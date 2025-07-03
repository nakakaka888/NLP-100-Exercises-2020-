import torch
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib
import numpy as np 
import time
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


test_x = torch.load('./section8/tensor_test.pth')
test_y = pd.read_csv('./section8/test_y.txt')
test_y = torch.tensor(test_y['CATEGORY'].values)


train_dataset = MyDataset(train_x,train_y)
test_dataset  = MyDataset(test_x,test_y)


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
batch_list = [1, 2, 4, 8, 16, 32]
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


def cal_Accuracy(dataloader, model):
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad(): #勾配計算の無効
        for X, y in dataloader:
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%\n")
    return correct



time_list = []
accuracy_list = []
for batch_size in batch_list:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    optimizer.zero_grad()
    nn.init.xavier_normal_(model.linear.weight)
    print(f"batch size: {batch_size}")
    start = time.time()
    for t in range(epochs):
        print(f"Epoch {t+1}\n----------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
    end = time.time()
    time_list.append(end - start)
    accuracy_list.append(cal_Accuracy(test_dataloader, model))


for batch, t, acc in zip(batch_list, time_list, accuracy_list):
    print(f"batch size:{batch},time:{t}, acc:{acc:>5f}")


"""
>実行結果
batch size:1,time:13.077002048492432, acc:0.882309
batch size:2,time:8.572794437408447, acc:0.877061
batch size:4,time:4.6156840324401855, acc:0.856822
batch size:8,time:2.5888900756835938, acc:0.820090
batch size:16,time:1.3897264003753662, acc:0.788606
batch size:32,time:0.932213544845581, acc:0.775862
"""