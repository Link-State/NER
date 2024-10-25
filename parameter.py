
import torch
import eojeol_etri_tokenizer.eojeol_tokenization as Tokenizer

SZ_TOKEN_VOCAB = len(Tokenizer.eojeol_BertTokenizer("./vocab.korean.rawtext.list", do_lower_case=False).vocab)
DIM_EMBEDDING = 768
NUM_HIDDEN_LAYER = 3
SZ_HIDDEN_STATE = 256
MSL = 128
BATCH_SIZE = 32
LIFE = 15
LABEL = ["[PAD]", "B-DT", "I-DT", "O", "B-LC", "I-LC", "B-OG", "I-OG", "B-PS", "I-PS", "B-TI", "I-TI", "X", "[CLS]", "[SEP]"]
LABEL2ID = dict()
for idx, label in enumerate(LABEL) :
    LABEL2ID[label] = idx
ENTITY_LABEL = [i for i in range(len(LABEL)) if len(LABEL[i])>=2 and LABEL[i][0]=="B" and LABEL[i][1]=="-"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
