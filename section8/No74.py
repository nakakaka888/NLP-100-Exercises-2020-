import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

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

test_x = torch.load('./section8/tensor_test.pth')
test_y = pd.read_csv('./section8/test_y.txt')
test_y = torch.tensor(test_y['CATEGORY'].values)


train_dataset = MyDataset(train_x,train_y)
train_dataloader = DataLoader(train_dataset, batch_size=32)

test_dataset = MyDataset(test_x,test_y)
test_dataloader = DataLoader(test_dataset, batch_size=32)


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
batch_size = 32
epochs = 10

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        #予測と損失
        pred = model(X)
        loss = loss_fn(pred, y)

        #バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def cal_Accuracy(dataloader, model):
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad(): #勾配計算の無効
        for X, y in dataloader:
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%\n")



for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)

print("train data\n----------------------------")
cal_Accuracy(train_dataloader, model)
print("test data\n----------------------------")
cal_Accuracy(test_dataloader, model)

"""
>実行結果
Epoch 7
----------------------------
loss: 0.729636  [    0/10672]
loss: 0.761369  [ 3200/10672]
loss: 0.721381  [ 6400/10672]
loss: 1.033894  [ 9600/10672]
Epoch 8
----------------------------
loss: 0.702607  [    0/10672]
loss: 0.727270  [ 3200/10672]
loss: 0.693535  [ 6400/10672]
loss: 1.005385  [ 9600/10672]
Epoch 9
----------------------------
loss: 0.679674  [    0/10672]
loss: 0.697605  [ 3200/10672]
loss: 0.669391  [ 6400/10672]
loss: 0.979524  [ 9600/10672]
Epoch 10
----------------------------
loss: 0.659850  [    0/10672]
loss: 0.671438  [ 3200/10672]
loss: 0.648156  [ 6400/10672]
loss: 0.955905  [ 9600/10672]
train data
----------------------------
Accuracy: 78.0%

test data
----------------------------
Accuracy: 77.4%
"""