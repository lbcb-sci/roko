import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torch.nn.init as init
import math


IN_SIZE = 600
HIDDEN_SIZE = 128
NUM_LAYERS = 2

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def gru_init(gru):
    stdv = math.sqrt(2.0 / gru.hidden_size)
    for param in gru.parameters():
        if len(param.shape) >= 2:
            init.orthogonal_(param.data)
        else:
            init.normal_(param.data)


class RNN(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, dropout=0.2):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(12, 50)  # B x R x S x E

        self.fc1 = nn.Linear(200, 100)
        self.do1 = nn.Dropout(0.2)
        #self.do1 = nn.Dropout(0.2, inplace=True)

        self.fc2 = nn.Linear(100, 12)
        self.do2 = nn.Dropout(0.2)
        #self.do2 = nn.Dropout(0.2, inplace=True)

        #self.conv1 = nn.Conv2d(10, 32, kernel_size=(1, 3), padding=(0, 1))
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        #self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1))

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #self.gru = nn.GRU(in_size, hidden_size, num_layers=num_layers,
                          #batch_first=True, bidirectional=True)
        #gru_init(self.gru)

        self.pos_encoder = PositionalEncoding(in_size, 0.2)

        encoder_layer = nn.TransformerEncoderLayer(in_size, 8, 4*in_size, 0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, 3)

        self.fc4 = nn.Linear(in_size, 5)
        #self.fc5 = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        t = None
        X = x

        for i in range(16):
            p = torch.randperm(200)
            x = X[:, p]

            x = self.embedding(x)
            x = x.permute((2, 0, 3, 1))  # S x B x E x R

            x = F.relu(self.fc1(x))
            x = self.do1(x)

            x = F.relu(self.fc2(x))
            x = self.do2(x)

        # x = torch.sum(x, dim=2)  # S x B x IN_SIZE
            x = x.reshape(90, -1, IN_SIZE)

            t = x if t is None else t + x

        t = t / 16.

        t = self.pos_encoder(t)

        t = self.encoder(t)

        return self.fc4(t).permute(1, 2, 0)

