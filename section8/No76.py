import torch
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib
import numpy as np 
matplotlib.use('Agg') # -----(1)


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx] , self.y[idx]


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(300, 4)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        logits = self.linear(x)
        return logits


train_x = torch.load('./section8/tensor_train.pth')
train_y = pd.read_csv('./section8/train_y.txt')
train_y = torch.tensor(train_y['CATEGORY'].values)

train_dataset = MyDataset(train_x,train_y)
train_dataloader = DataLoader(train_dataset, batch_size=8)

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
batch_size = 8
epochs = 10

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer,epoch):
    size = len(dataloader.dataset)
    total_loss, correct, total = 0, 0, 0
    for X, y in dataloader:
        total += 1

        #予測と損失
        pred = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        #バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #正解率,損失
    correct /= size
    total_loss /= total

    checkpoint = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    }
    
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%,  loss: {total_loss:>8f} \n")
    return checkpoint

for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    chack_point = train_loop(train_dataloader, model, loss_fn, optimizer,t)
    torch.save(chack_point, f"./section8/chack_point{t+1}.pth")

"""
Epoch 1
----------------------------
Train Error:
 Accuracy: 75.2%,  loss: 0.994747

Epoch 2
----------------------------
Train Error:
 Accuracy: 77.6%,  loss: 0.790985

Epoch 3
----------------------------
Train Error:
 Accuracy: 77.8%,  loss: 0.703426

Epoch 4
----------------------------
Train Error:
 Accuracy: 78.0%,  loss: 0.647881

Epoch 5
----------------------------
Train Error:
 Accuracy: 78.5%,  loss: 0.607701

Epoch 6
----------------------------
Train Error:
 Accuracy: 79.3%,  loss: 0.576480

Epoch 7
----------------------------
Train Error:
 Accuracy: 80.4%,  loss: 0.551152

Epoch 8
----------------------------
Train Error:
 Accuracy: 81.3%,  loss: 0.530015

Epoch 9
----------------------------
Train Error:
 Accuracy: 82.2%,  loss: 0.512023

Epoch 10
----------------------------
Train Error:
 Accuracy: 83.0%,  loss: 0.496481

"""