# 한국어 문장에서 개체명을 인식하는 시스템 개발
### [2024 2학기 자연어처리 과제2]

### 개발 기간
> 2024.10.06 ~ 2024.10.15

### 개발 환경
> Python 3.12.6 (venv)<br>
> Pytorch 2.4.1 + CUDA 12.4<br>
> RTX4050 Laptop<br>

### 설명
+ 동기
    + 자연어처리 수업 과제
+ 기획
    + 학습 데이터는 ETRI가 구축한 한국어 개체명 인식용 tagged corpus를 사용한다.
    + 어절 및 토큰화는 eojeol_etri_tokenizer 모듈을 사용한다.
    + RNN과 LSTM 두 종류의 인공신경망을 이용하여 시스템을 개발한다.
    + 훈련:검증:테스트는 6:2:2 비율로 나눈다.
    + 사용자가 문장을 입력하여 개체명을 인식하고 태깅하여 출력한다.
    + 모델 개발 후 테스트 입력으로 넣을 문장을 ChatGPT를 통해 얻는다.

### 1) RNN
#### 1-1) 학습-검증 오차 그래프
![image03](https://github.com/user-attachments/assets/b6b4758b-fa1a-4eb1-8df8-370dce0d74cc)

#### 1-2) 성능지표
![image05](https://github.com/user-attachments/assets/b3921824-73a6-4c50-9390-cb5ceeaca399)


#### 1-3) 입력 결과
![image01](https://github.com/user-attachments/assets/3b46595e-0f01-4eab-ba45-4a4e50f3faf7)

<br>

### 2) LSTM
#### 2-1) 학습-검증 오차 그래프
![image04](https://github.com/user-attachments/assets/bff01beb-693e-4cda-9269-236cfb381283)

#### 2-2) 성능지표
![image06](https://github.com/user-attachments/assets/cf8a64ac-986b-43d6-8531-8243b17173f9)

#### 2-3) 입력 결과
![image02](https://github.com/user-attachments/assets/2b53922c-3c0b-41f2-b6ea-e6e0f1511939)
<br>

