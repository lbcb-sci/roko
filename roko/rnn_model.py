import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torch.nn.init as init
import math


IN_SIZE = 256
HIDDEN_SIZE = 128
NUM_LAYERS = 2


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

        self.embedding = nn.Embedding(12, 10)  # B x R x S x E
        
        self.conv1 = nn.Conv2d(10, 32, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1))

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(in_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True)
        gru_init(self.gru)

        self.fc4 = nn.Linear(2 * hidden_size, 5)

    def forward(self, x):
        x = self.do(self.embedding(x))
        x = x.permute((0, 3, 1, 2))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.sum(x, dim=2)
        x = x.permute((0, 2, 1))

        x, _ = self.gru(x)

        return self.fc4(x)

