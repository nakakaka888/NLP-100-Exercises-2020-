import torch
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib
import numpy as np 
matplotlib.use('Agg') # -----(1)

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

valid_x = torch.load('./section8/tensor_valid.pth')
valid_y = pd.read_csv('./section8/valid_y.txt')
valid_y = torch.tensor(valid_y['CATEGORY'].values)


train_dataset = MyDataset(train_x,train_y)
train_dataloader = DataLoader(train_dataset, batch_size=8)

valid_dataset = MyDataset(valid_x,valid_y)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
batch_size = 8
epochs = 30

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
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
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%,  loss: {total_loss:>8f} \n")
    return correct, total_loss



def valid_loop(dataloader, model):
    size = len(dataloader.dataset)
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad(): #勾配計算の無効
        for X, y in dataloader:
            total += 1
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_loss += loss_fn(pred, y)

    #正解率,損失
    correct /= size
    total_loss /= total
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, loss: {total_loss:>8f} \n")
    return correct, total_loss


result_acc = []
result_loss = []
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train_acc, train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    valid_acc, valid_loss = valid_loop(valid_dataloader, model)
    result_acc.append([train_acc, valid_acc])
    result_loss.append([train_loss, valid_loss])    

fig = plt.figure(figsize = (10,6))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(np.array(result_acc).T[0], label = "train")
ax1.plot(np.array(result_acc).T[1], label = "valid")

ax2.plot(np.array(result_loss).T[0], label = "train")
ax2.plot(np.array(result_loss).T[1], label = "valid")

ax1.set_xlabel("epoch")
ax2.set_xlabel("epoch")

ax1.set_ylabel("accuracy")
ax2.set_ylabel("loss")

ax1.legend(loc = 'upper left') 
ax2.legend(loc = 'upper left') 

plt.savefig('./section8/No75.png')