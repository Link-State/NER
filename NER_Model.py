
import torch
from torch import nn
from parameter import *

# RNN
class RNN_Model(nn.Module) :
    def __init__(self) :
        super(RNN_Model, self).__init__()
        self.embedding = nn.Embedding(SZ_TOKEN_VOCAB, DIM_EMBEDDING, padding_idx=0)
        self.RNN = nn.RNN(
            input_size=DIM_EMBEDDING,
            hidden_size=SZ_HIDDEN_STATE,
            num_layers=NUM_HIDDEN_LAYER,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.Linear = nn.Linear(SZ_HIDDEN_STATE*2, len(LABEL), bias=True)

    
    def forward(self, X, batch_size) :
        vtr = self.embedding(X)

        hidden = torch.zeros(NUM_HIDDEN_LAYER*2, batch_size, SZ_HIDDEN_STATE, requires_grad=True).to(DEVICE)
        y, h = self.RNN(vtr, hidden)

        y = self.Linear(y)
        return y


# LSTM
class LSTM_Model(nn.Module) :
    def __init__(self) :
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(SZ_TOKEN_VOCAB, DIM_EMBEDDING, padding_idx=0)
        self.LSTM = nn.LSTM(
            input_size=DIM_EMBEDDING,
            hidden_size=SZ_HIDDEN_STATE,
            num_layers=NUM_HIDDEN_LAYER,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.Linear = nn.Linear(SZ_HIDDEN_STATE*2, len(LABEL), bias=True)
    
    def forward(self, X, batch_size) :
        vtr = self.embedding(X)

        hidden = torch.zeros(NUM_HIDDEN_LAYER*2, batch_size, SZ_HIDDEN_STATE, requires_grad=True).to(DEVICE)
        cell = torch.zeros(NUM_HIDDEN_LAYER*2, batch_size, SZ_HIDDEN_STATE, requires_grad=True).to(DEVICE)
        y, h = self.LSTM(vtr, (hidden, cell))

        y = self.Linear(y)
        return y
