import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.optim as optim
import torch.utils.data as torch_data
import torchaudio
import tqdm.notebook as tqdm
import dataset
import urllib
import random
from ecapa_tdnn import *
import warnings
from IPython.display import clear_output
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Res2DilatedConv1D(nn.Module):
    def __init__(self, kernel_size: int, dilation: int, C: int, scale: int, padding: int):
        super().__init__()
        self.k = kernel_size
        self.d = dilation
        self.scale = scale
        self.nums = scale
        self.width = C // scale
        self.p = padding

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, self.k, dilation=self.d, padding=self.p))
            self.bns.append(nn.BatchNorm1d(self.width))

        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, X):
        outputs = []
        splits = torch.split(X, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                split = splits[i]
            else:
                split += splits[i]

            split = self.convs[i](split)
            split = self.bns[i](F.relu(split))
            outputs.append(split)
        return torch.cat(outputs, dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels: int, s: int):
        super().__init__()
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, X):
        output = torch.mean(X, dim=2)
        output = F.relu(self.linear1(output))
        output = torch.sigmoid(self.linear2(output))
        output = X * output.unsqueeze(2)
        return output

class SERes2Block(nn.Module):
    def __init__(self, kernel_size: int, dilation: int, C: int, scale: int, padding: int):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(C)
        )
        self.res2dilated_block = Res2DilatedConv1D(kernel_size, dilation, C, scale, padding)
        self.third_block = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(C)
        )
        self.se_block = SEBlock(C, 2)

    def forward(self, X):
        X = self.first_block(X)
        X = self.res2dilated_block(X)
        X = self.third_block(X)
        X = self.se_block(X)
        return X

class AttentiveStatPooling(nn.Module):
    def __init__(self, emb_size: int, hidden: int):
        super().__init__()
        self.emb_size = emb_size
        self.Q = nn.Conv1d(emb_size, hidden, kernel_size=1)
        self.K = nn.Conv1d(hidden, emb_size, kernel_size=1)

    def forward(self, X):
        out = torch.tanh(self.Q(X))
        attention_weights = torch.softmax(self.K(out), dim=-1)
        weighted_mean = torch.sum(attention_weights * X, dim=-1)
        weighted_std = torch.sqrt((torch.sum(attention_weights * X * X, dim=-1) - weighted_mean * weighted_mean).clamp(min=1e-9))
        return torch.cat([weighted_mean, weighted_std], dim=1)

class EcapaTDNN(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, C: int):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.Conv1d(input_shape, out_channels=C, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(C),
        )
        self.second_block = SERes2Block(kernel_size=3, dilation=2, scale=8, padding=2, C=C)
        self.third_block = SERes2Block(kernel_size=3, dilation=3, scale=8, padding=3, C=C)
        self.fourth_block = SERes2Block(kernel_size=3, dilation=4, scale=8, padding=4, C=C)

        self.fifth_block = nn.Sequential(
            nn.Conv1d(C * 3, out_channels=1536, kernel_size=1),
            nn.ReLU()
        )

        self.attention_block = AttentiveStatPooling(emb_size=1536, hidden=128)
        self.bn1 = nn.BatchNorm1d(3072)

        self.linear = nn.Linear(3072, output_shape)
        self.bn2 = nn.BatchNorm1d(output_shape)

    def forward(self, X):

        out1 = self.first_block(X)
        out2 = self.second_block(out1) + out1
        out3 = self.third_block(out2) + out2 + out1
        out4 = self.fourth_block(out3) + out3 + out2 + out1

        out = torch.cat([out2, out3, out4], dim=1)
        out = self.fifth_block(out)

        emb = self.bn1(self.attention_block(out))
        out = self.bn2(self.linear(emb))
        return out, emb