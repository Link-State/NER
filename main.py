import os

if not os.path.exists("./eojeol_etri_tokenizer") :
    print("eojeol_etri_tokenizer 폴더(모듈)가 같은 디렉토리 내 존재하지 않습니다.")
    exit(0)

import re
import time
from datetime import datetime
import numpy as np
import eojeol_etri_tokenizer.eojeol_tokenization as Tokenizer
import eojeol_etri_tokenizer.file_utils as TokenUtils
from NER_Model import RNN_Model, LSTM_Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler
from parameter import *
from matplotlib import pyplot as plt


# 로그 출력
def log(msg="", path="./log.txt", output=True) :
    log_file = open(path, "a", encoding="utf-8")
    log_file.write(str(msg) + "\n")
    log_file.close()
    if output :
        print(msg)
    return


# 문장 분석
def eojeol() :
    if not os.path.exists("./NER_tagged_corpus_ETRI_exobrain_team.txt") :
        log("NER_tagged_corpus_ETRI_exobrain_team.txt파일이 같은 디렉토리 내 존재하지 않습니다.")
        return False

    fp = open("./NER_tagged_corpus_ETRI_exobrain_team.txt", "r", encoding="utf-8")

    regex = re.compile(r'<[^:<>]*:[^:<>]*>')

    for line in fp :
        _line = line.strip()
        index = dict()
        iterator = regex.finditer(_line)
        for match in iterator :
            tag = match.group()[1:-1].split(":")
            index[match.start()] = [match.end(), tag[0], tag[1]]
        
        start = 0
        end = 0
        eojeol_label = open("./ner_eojeol_label_per_line.txt", "a", encoding="utf-8")
        while end < len(_line) :
            if _line[end] == " " :
                word = _line[start:end].strip()
                if word != "" :
                    eojeol_label.write(word + "\t" + "O\n")
                start = end + 1
                end = start
            
            if end in index :
                if start < end :
                    word = _line[start:end].strip()
                    if word != "" :
                        eojeol_label.write(word + "\t" + "O\n")
                
                splited_tag = index[end][1].split()
                eojeol_label.write(splited_tag[0].strip() + "\t" + "B-" + index[end][2] + "\n")
                for i in range(1, len(splited_tag)) :
                    eojeol_label.write(splited_tag[i].strip() + "\t" + "I-" + index[end][2] + "\n")
                
                start = index[end][0] + 1
                end = start
            else :
                end += 1
                if end == len(_line) and end >= start :
                    word = _line[start:].strip()
                    if word != "" :
                        eojeol_label.write(word + "\t" + "O\n")
        eojeol_label.write("\n")
        eojeol_label.close()    
    fp.close()
    return True


# 토큰화
def tokenizer() :
    eojeol_tokenizer = Tokenizer.eojeol_BertTokenizer("./vocab.korean.rawtext.list", do_lower_case=False)
    fp = open("./ner_eojeol_label_per_line.txt", "r", encoding="utf-8")
    while True :
        e_sent = fp.readline()
        if e_sent == '' :
            break
        if len(e_sent) < 2 :
            label_str_file = open("./ner_token_label_per_line.txt", "a", encoding="utf-8")
            label_id_file = open("./ner_token_id_label_id_per_line.txt", "a", encoding="utf-8")

            label_str_file.write("\n")
            label_id_file.write("\n")

            label_str_file.close()
            label_id_file.close()
            continue
        
        splited_e_sent = e_sent.split()
        if len(splited_e_sent) < 2 :
            continue

        token = eojeol_tokenizer.tokenize(splited_e_sent[0])
        token_id = eojeol_tokenizer.convert_tokens_to_ids(token)

        labeling(tokens=token, tk_ids=token_id, tag=splited_e_sent[1])

    fp.close()
    return


# 레이블링
def labeling(tokens=list(), tk_ids=list(), tag="") :
    label_str_file = open("./ner_token_label_per_line.txt", "a", encoding="utf-8")
    label_id_file = open("./ner_token_id_label_id_per_line.txt", "a", encoding="utf-8")

    label_str_file.write(tokens[0] + "\t" + tag + "\n")
    label_id_file.write(str(tk_ids[0]) + "\t" + str(LABEL2ID[tag]) + "\n")
    for i in range(1, len(tokens)) :
        label_str_file.write(tokens[i] + "\t" + "X\n")
        label_id_file.write(str(tk_ids[i]) + "\t" + str(LABEL2ID["X"]) + "\n")

    label_str_file.close()
    label_id_file.close()
    return


# 훈련 예제 생성 및 반환
def getTrainExample(path="./ner_token_id_label_id_per_line.txt") :
    ids_list = []

    f = open(path, "r", encoding="utf-8-sig")
    for line in f :
        ids_list.append(line[:-1])
    f.close()

    x = []
    y = []
    temp_x = []
    temp_y = []

    for ids in ids_list :
        if len(ids) == 0 :
            if len(temp_x) < MSL :
                temp_x.extend([LABEL2ID[LABEL[0]]] * (MSL - len(temp_x)))
                temp_y.extend([LABEL2ID[LABEL[0]]] * (MSL - len(temp_y)))
            elif len(temp_x) > MSL :
                temp_x = temp_x[:MSL]
                temp_y = temp_y[:MSL]

            assert(len(temp_x) == MSL)
            assert(len(temp_y) == MSL)

            x.append(temp_x)
            y.append(temp_y)

            temp_x = list()
            temp_y = list()
        else :
            ids = ids.split('\t')
            temp_x.extend([int(ids[0])])
            temp_y.extend([int(ids[1])])
    x = np.array(x, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    return x, y


# 학습 시작
def startLearning(x=np.array([0]), y=np.array([0]), model=RNN_Model(), opt="AdamW") :
    num_examples = x.shape[0]
    num_tra = int(0.6 * num_examples)
    num_tes = int(0.8 * num_examples)
    num_val = num_tes - num_tra

    # 학습용 데이터 셋
    train_x = np.zeros((num_tra, MSL), np.int32)
    train_y = np.zeros((num_tra, MSL), np.int32)

    train_x[:, :] = x[:num_tra, :]
    train_y[:, :] = y[:num_tra, :]

    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    train_data = TensorDataset(train_x, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # 검증용 데이터 셋
    valid_x = np.zeros((num_val, MSL), np.int32)
    valid_y = np.zeros((num_val, MSL), np.int32)

    valid_x[:, :] = x[num_tra:num_tes, :]
    valid_y[:, :] = y[num_tra:num_tes, :]

    valid_x = torch.LongTensor(valid_x)
    valid_y = torch.LongTensor(valid_y)
    valid_data = TensorDataset(valid_x, valid_y)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)

    # optimizer 선택
    optimizer = None
    if opt == "SGD" :
        optimizer = optim.SGD(model.parameters(), lr=1e-5, weight_decay=0.001)
    elif opt == "Adam" :
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)
    else :
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, .999), eps=1e-08, weight_decay=0.01)

    modelName = str(type(model)).replace("class", "", 1).replace(".", "-").replace("<", "").replace(">", "").replace(" ", "").replace("'", "")

    train_loss = list()
    valid_loss = list()

    minimum_loss = -1.0
    best_epoch = -1
    epoch = 0
    life = 0
    while True :
        model, avg_loss = trainModel(train_dataloader, optimizer, model)
        train_loss.append(avg_loss)

        log(f"\tEpoch : {epoch} is finished. Avg_loss = {avg_loss}")

        avg_loss = validateModel(valid_dataloader, model)
        valid_loss.append(avg_loss)

        if  minimum_loss < 0 or minimum_loss > avg_loss :
            minimum_loss = avg_loss
            best_epoch = epoch + 1
            life = 0
            torch.save(model.state_dict(), f"./{modelName}_{opt}.pth")
        elif life >= LIFE :
            break

        epoch += 1
        life += 1

    log(f"\tBest epoch : {best_epoch}")
    
    x_axis = range(epoch + 1)
    plt.plot(x_axis, train_loss, label="training")
    plt.plot(x_axis, valid_loss, label="validation")
    plt.axvline(x=best_epoch-1, ymin=0, ymax=1, linestyle="--", label="best epoch point")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("./" + modelName + "_" + opt + ".png")
    
    return model


# 모델 학습
def trainModel(dataloader, optimizer, model) :
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    total_loss = 0.0
    model.train()
    for j, batch in enumerate(dataloader) :
        batch = tuple(r.to(DEVICE) for r in batch)
        X, Y = batch
        optimizer.zero_grad()
        model.zero_grad()
        preds = model(X, X.shape[0])
        preds_tr = torch.transpose(preds, 1, 2)

        loss = loss_fn(preds_tr, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / ((j+1) * BATCH_SIZE)
    return (model, avg_loss)


# 모델 검증
def validateModel(dataloader, model) :
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    total_loss = 0.0
    model.eval()
    with torch.no_grad() :
        for k, batch in enumerate(dataloader) :
            batch = tuple(r.to(DEVICE) for r in batch)
            X, Y = batch
            preds = model(X, X.shape[0])
            preds_tr = torch.transpose(preds, 1, 2)

            loss = loss_fn(preds_tr, Y).item()
            total_loss += loss
        avg_loss = total_loss / ((k+1) * BATCH_SIZE)
    return avg_loss


# 모델 테스트
def testModel(x=np.array([0]), y=np.array([0]), model=RNN_Model()) :
    num_examples = x.shape[0]
    num_tes = int(0.8 * num_examples)

    test_x = np.zeros((num_examples-num_tes, MSL), np.int32)
    test_y = np.zeros((num_examples-num_tes, MSL), np.int32)

    test_x[:, :] = x[num_tes:, :]
    test_y[:, :] = y[num_tes:, :]

    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)
    test_data = TensorDataset(test_x, test_y)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)
    
    softmax_fx = torch.nn.Softmax(dim=2)

    total_TN, total_FP, total_FN, total_TP = 0, 0, 0, 0
    model.eval()
    with torch.no_grad() :
        for k, batch in enumerate(test_dataloader) :
            batch = tuple(r.to(DEVICE) for r in batch)
            X, Y = batch
            preds = model(X, X.shape[0])
            preds = softmax_fx(preds)
            pred_label = torch.argmax(preds, dim=2)

            for i in range(len(Y)) :
                target_label_seq = Y[i]
                pred_label_seq = pred_label[i]
                tn, fp, fn, tp = 0, 0, 0, 0
                for j in range(MSL) :
                    # NE인데, 정답을 정확히 맞춘 경우 (True Positive)
                    if target_label_seq[j] in ENTITY_LABEL and target_label_seq[j] == pred_label_seq[j] :
                        tp += 1
                    # NE가 아닌데, 실제로도 NE가 아니었던 경우 (True Negative)
                    elif target_label_seq[j] not in ENTITY_LABEL and pred_label_seq[j] not in ENTITY_LABEL :
                        tn += 1
                    # NE인데, 정답을 못 맞춘 경우 (False Positive)
                    elif target_label_seq[j] in ENTITY_LABEL and target_label_seq[j] != pred_label_seq[j] :
                        fp += 1
                    # NE가 아닌데, 실제로는 NE였던 경우 (False Negative)
                    elif target_label_seq[j] not in ENTITY_LABEL and pred_label_seq[j] in ENTITY_LABEL :
                        fn += 1
                total_TP += tp
                total_TN += tn
                total_FP += fp
                total_FN += fn

        accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
        recall = total_TP / (total_TP + total_FN)
        precision = total_TP / (total_TP + total_FP)
        f1_score = (2 * recall * precision) / (recall + precision)
        log(f"\tAccuracy : {accuracy}")
        log(f"\tRecall : {recall}")
        log(f"\tPrecision : {precision}")
        log(f"\tF1-Score : {f1_score}")

    return


# 문장 -> 토큰
def getToken(line="") :
    words = line.split()

    result = []
    temp = []
    tokens = []
    eojeol_tokenizer = Tokenizer.eojeol_BertTokenizer("./vocab.korean.rawtext.list", do_lower_case=False)

    for word in words :
        if len(word) <= 0 :
            continue
        token = eojeol_tokenizer.tokenize(word)
        token_id = eojeol_tokenizer.convert_tokens_to_ids(token)
        temp.extend(token_id)
        tokens.extend(token)
    
    if len(temp) < MSL :
        temp.extend([LABEL2ID[LABEL[0]]] * (MSL - len(temp)))
    elif len(temp) > MSL :
        temp = temp[:MSL]
    result.append(temp)
    
    result = np.array(result, dtype=np.int32)
    return result, tokens


# 토큰열 -> 훈련예제
def convertExample(line="") :
    x, tokens = getToken(line)
    num_examples = x.shape[0]

    train_x = np.zeros((num_examples, MSL), np.int32)
    train_x[:, :] = x[:num_examples, :]

    train_x = torch.LongTensor(train_x)
    train_data = TensorDataset(train_x)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    
    X = None
    for r in train_dataloader :
        X = r[0].to(DEVICE)
    
    return X, tokens


# 예측결과 -> 태깅된 개체
def findEntity(tokens, label) :
    word_list = list()
    word = ""
    tag = ""
    for i in range(len(tokens)) :
        if label[i].item() in ENTITY_LABEL :
            if word == "" :
                word = tokens[i]
                tag = LABEL[label[i].item()][2:]
            else :
                word = word.replace("_", "").strip()
                word_list.append([word, tag])
                word = tokens[i]
                tag = LABEL[label[i].item()][2:]
            if tokens[i][-1] == "_" and label[i+1].item() not in [2, 5, 7, 9, 11] :
                word = word.replace("_", " ").strip()
                word_list.append([word, tag])
                word = ""
                tag = ""
        elif word != "" and len(tokens[i]) >= 1 and tokens[i][-1] == "_"  :
            word += tokens[i]
            if label[i+1].item() not in [2, 5, 7, 9, 11] :
                word = word.replace("_", " ").strip()
                word_list.append([word, tag])
                word = ""
                tag = ""
        elif word != "" :
            word += tokens[i]
    
    if word != "" :
        word = word.replace("_", " ").strip()
        word_list.append([word, tag])

    return word_list


# Main 함수
def main() :
    log(msg="\n------------------------" + str(datetime.now()) + "------------------------\n", output=False)
    log(f"DEVICE : {DEVICE}\n")

    # 저장된 모델 불러오기
    model_files = list()
    file_list = os.listdir("./")
    for file_name in file_list :
        if os.path.splitext(file_name)[1] == ".pth" :
            model_files.append(file_name)
    
    model = None
    selectedModelFile = ""
    if len(model_files) > 0 :
        log("------------------------")
        for i in range(len(model_files)) :
            log(f"[{i+1}]   {model_files[i]}")
        log("[N]   사용 안함")
        log("------------------------")
        cmd = input("불러올 모델파일 번호 입력   >> ").upper()
        log(f"불러올 모델파일 번호 입력   >> {cmd}", output=False)
        log("")

        if cmd.isnumeric() and len(model_files) >= int(cmd) and 0 < int(cmd) :
            selectedModelFile = model_files[int(cmd)-1]

    if selectedModelFile != "" :
        try :
            model = RNN_Model()
            model.load_state_dict(torch.load("./" + selectedModelFile))
            model.to(DEVICE)
        except :
            model = LSTM_Model()
            model.load_state_dict(torch.load("./" + selectedModelFile))
            model.to(DEVICE)
        log("")
    else :
        # 모델 선택
        log("------------------------")
        log("1.RNN(기본)\t2.LSTM")
        log("------------------------")
        cmd = input("모델번호 입력 >> ")
        log(f"모델 번호 입력 >> {cmd}", output=False)
        log("\n")
        if cmd == "2" :
            model = LSTM_Model()
        else :
            model = RNN_Model()

        # 옵티마이저 선택
        optimizer = "AdamW"
        log("-----------------------------")
        log(f"1.SGD\t2.Adam\t3.AdamW(기본)")
        log("-----------------------------")
        cmd = input("옵티마이저 번호 입력 >> ")
        log(f"옵티마이저 번호 입력 >> {cmd}", output=False)
        log("\n")
        if cmd == "1" :
            optimizer = "SGD"
        elif cmd == "2" :
            optimizer = "Adam"

        # 문장 분석한 파일 생성
        if not os.path.exists("./ner_eojeol_label_per_line.txt") :
            log("문장 분석 중")
            start_time = time.perf_counter()
            isSuccess = eojeol()
            if not isSuccess :
                return
            end_time = time.perf_counter()
            elapse = end_time - start_time
            log(f"└분석 완료 ({elapse:.3f} sec)\n")

        # 각 문장 토큰화 및 레이블링
        if not os.path.exists("./ner_token_label_per_line.txt") and not os.path.exists("./ner_token_id_label_id_per_line.txt") :
            log("토큰화 중")
            start_time = time.perf_counter()
            tokenizer()
            end_time = time.perf_counter()
            elapse = end_time - start_time
            log(f"└토큰화 완료 ({elapse:.3f} sec)\n")
    
        # 학습 예제 생성
        log("학습 예제 생성 중")
        start_time = time.perf_counter()
        x, y = getTrainExample()
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log(f"└학습 예제 생성 완료 ({elapse:.3f} sec)\n")

        # 모델 훈련
        log("훈련 중")
        start_time = time.perf_counter()
        model = startLearning(x, y, model, opt=optimizer)
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log(f"└훈련 완료 ({elapse:.3f} sec)\n")

        # 모델 테스팅
        log("테스팅 중")
        start_time = time.perf_counter()
        testModel(x, y, model)
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log(f"└테스팅 완료 ({elapse:.3f} s)\n")

    # 실제 입력
    cmd = "dummy"
    while True :
        cmd = input("문장 입력 >> ")
        log(f"문장 입력 >> {cmd}", output=False)

        if cmd == "" :
            break
        user_x, tokens = convertExample(cmd)

        pred = model(user_x, user_x.shape[0])
        softmax_fn = torch.nn.Softmax(dim=2)
        pred = softmax_fn(pred)
        pred_label = torch.argmax(pred, dim=2)

        tagged_list = findEntity(tokens, pred_label[0])
        for t in tagged_list :
            log(f"{t[0]} : {t[1]}")
        if len(tagged_list) <= 0 :
            log("결과가 없습니다.")
        log("\n")
    return


if __name__ == "__main__" :
    main()
