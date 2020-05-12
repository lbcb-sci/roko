import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torch.nn.init as init
import math


IN_SIZE = 500
HIDDEN_SIZE = 128
NUM_LAYERS = 3


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

        self.embedding = nn.Embedding(12, 50)
        self.do = nn.Dropout(dropout)

        self.fc1 = nn.Linear(200, 100)
        self.do1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(100, 10)
        self.do2 = nn.Dropout(dropout)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(in_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        gru_init(self.gru)

        self.fc4 = nn.Linear(2 * hidden_size, 5)

    def forward(self, x):
        x = self.do(self.embedding(x))
        x = x.permute((0, 2, 3, 1))

        x = F.relu(self.fc1(x))
        x = self.do1(x)

        x = F.relu(self.fc2(x))
        x = self.do2(x)

        x = x.reshape(-1, 90, IN_SIZE)

        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        return self.fc4(x)

