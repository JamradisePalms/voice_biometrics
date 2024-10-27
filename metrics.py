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
from dataset import Dataset
import urllib
#from adan_pytorch import Adan
import random
from ecapa_tdnn import *
import warnings
from IPython.display import clear_output
warnings.filterwarnings("ignore", category=DeprecationWarning)


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    max_length = max(item[0].shape[1] for item in batch)
    X = torch.zeros((len(batch), batch[0][0].shape[0], max_length))
    for idx, item in enumerate(batch):
        X[idx, :, :item[0].shape[1]] = item[0]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long).reshape(len(batch), 1)
    pathes = [item[2] for item in batch]
    return (X, targets, pathes)

def cosine_similarity(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

def best_eer(data):
    full = sorted(data, key=lambda x: (x[0], -x[1]))
    pos = len([item for item in full if item[1] == 1])
    neg = len(full) - pos
    cur_pos = pos
    cur_neg = 0
    best_eer = 1
    for _, label in full:
        if label == 1:
            cur_pos -= 1
        else:
            cur_neg += 1
        cur_eer = max((pos - cur_pos) / pos, (neg - cur_neg) / neg)
        best_eer = min(best_eer, cur_eer)
    return best_eer

def calc_eval_score(model: nn.Module, batch_size: int = 128):
    loader = torch_data.DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=LOADER_WORKERS,
    )
    items = {}
    target_scores = []
    with torch.no_grad():
        for X, _, pathes in tqdm.tqdm(loader):
            _, embds = model.forward(X.to(DEVICE))
            embds = embds.cpu().data.numpy().reshape(X.shape[0], -1)
            for embd, path in zip(embds, pathes):
                items[path] = embd
    for item1, item2, target in test_targets:
        target_scores.append((cosine_similarity(items[item1], items[item2]), target))
    return best_eer(target_scores)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATADIR = ''
FEATS = 80
LOADER_WORKERS = 4
transform = torchaudio.transforms.MelSpectrogram(n_mels=FEATS)
trainset = Dataset(os.path.join('', 'voxceleb_train'), transform)
testset = Dataset(os.path.join('', 'voxceleb_test'), transform)
test_targets = pd.read_csv(os.path.join('', 'target.csv')).values.tolist()

def ecapa_metrics():
    ecapa_model = EcapaTDNN(FEATS, trainset.speakers(), 512).to(DEVICE)
    ecapa_model.load_state_dict(torch.load('ecapa_triplet.pt', weights_only=True))
    return calc_eval_score(ecapa_model, batch_size=64)