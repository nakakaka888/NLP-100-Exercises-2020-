import torch
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np 
import time 


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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        nn.init.xavier_normal_(self.linear_relu_stack[0].weight)
        nn.init.xavier_normal_(self.linear_relu_stack[2].weight)
        nn.init.xavier_normal_(self.linear_relu_stack[4].weight)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_x = torch.load('./section8/tensor_train.pth').to(device)
train_y = pd.read_csv('./section8/train_y.txt')
train_y = torch.tensor(train_y['CATEGORY'].values).to(device)

test_x = torch.load('./section8/tensor_test.pth').to(device)
test_y = pd.read_csv('./section8/test_y.txt')
test_y = torch.tensor(test_y['CATEGORY'].values).to(device)


train_dataset = MyDataset(train_x,train_y)
train_dataloader = DataLoader(train_dataset, batch_size=8)

test_dataset  = MyDataset(test_x,test_y)
test_dataloader = DataLoader(test_dataset, batch_size=8)

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
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



def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad(): #勾配計算の無効
        for X, y in dataloader:
            total += 1
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_loss += loss_fn(pred, y).item()

    #正解率,損失
    correct /= size
    total_loss /= total
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, loss: {total_loss:>8f} \n")

start = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model)
end = time.time()

print(f"time:{end - start}")


"""
Epoch 1
----------------------------
Train Error:
 Accuracy: 73.0%,  loss: 0.873284

Test Error:
 Accuracy: 77.9%, loss: 0.644612

Epoch 2
----------------------------
Train Error:
 Accuracy: 78.6%,  loss: 0.584839

Test Error:
 Accuracy: 78.4%, loss: 0.570290

Epoch 3
----------------------------
Train Error:
 Accuracy: 79.1%,  loss: 0.530406

Test Error:
 Accuracy: 78.6%, loss: 0.534676

Epoch 4
----------------------------
Train Error:
 Accuracy: 80.0%,  loss: 0.494815

Test Error:
 Accuracy: 81.9%, loss: 0.468451

Epoch 5
----------------------------
Train Error:
 Accuracy: 83.7%,  loss: 0.445214

Test Error:
 Accuracy: 83.6%, loss: 0.438289

Epoch 6
----------------------------
Train Error:
 Accuracy: 84.5%,  loss: 0.421346

Test Error:
 Accuracy: 84.1%, loss: 0.419130

Epoch 7
----------------------------
Train Error:
 Accuracy: 85.3%,  loss: 0.401661

Test Error:
 Accuracy: 85.1%, loss: 0.401847

Epoch 8
----------------------------
Train Error:
 Accuracy: 85.6%,  loss: 0.383178

Test Error:
 Accuracy: 85.5%, loss: 0.386211

Epoch 9
----------------------------
Train Error:
 Accuracy: 85.8%,  loss: 0.367620

Test Error:
 Accuracy: 85.5%, loss: 0.374407

Epoch 10
----------------------------
Train Error:
 Accuracy: 85.9%,  loss: 0.356192

Test Error:
 Accuracy: 85.7%, loss: 0.366044

Epoch 11
----------------------------
Train Error:
 Accuracy: 86.0%,  loss: 0.347354

Test Error:
 Accuracy: 85.7%, loss: 0.360202

Epoch 12
----------------------------
Train Error:
 Accuracy: 86.1%,  loss: 0.340231

Test Error:
 Accuracy: 85.4%, loss: 0.356062

Epoch 13
----------------------------
Train Error:
 Accuracy: 86.3%,  loss: 0.334301

Test Error:
 Accuracy: 85.7%, loss: 0.352963

Epoch 14
----------------------------
Train Error:
 Accuracy: 86.4%,  loss: 0.329221

Test Error:
 Accuracy: 86.0%, loss: 0.350593

Epoch 15
----------------------------
Train Error:
 Accuracy: 86.5%,  loss: 0.324706

Test Error:
 Accuracy: 86.1%, loss: 0.349107

Epoch 16
----------------------------
Train Error:
 Accuracy: 86.6%,  loss: 0.320759

Test Error:
 Accuracy: 86.2%, loss: 0.348102

Epoch 17
----------------------------
Train Error:
 Accuracy: 86.7%,  loss: 0.317098

Test Error:
 Accuracy: 86.4%, loss: 0.347445

Epoch 18
----------------------------
Train Error:
 Accuracy: 86.8%,  loss: 0.313657

Test Error:
 Accuracy: 86.5%, loss: 0.347111

Epoch 19
----------------------------
Train Error:
 Accuracy: 86.8%,  loss: 0.310497

Test Error:
 Accuracy: 86.6%, loss: 0.347163

Epoch 20
----------------------------
Train Error:
 Accuracy: 86.9%,  loss: 0.307461

Test Error:
 Accuracy: 86.5%, loss: 0.347588

Epoch 21
----------------------------
Train Error:
 Accuracy: 87.0%,  loss: 0.304632

Test Error:
 Accuracy: 86.6%, loss: 0.347930

Epoch 22
----------------------------
Train Error:
 Accuracy: 87.1%,  loss: 0.301926

Test Error:
 Accuracy: 86.7%, loss: 0.348485

Epoch 23
----------------------------
Train Error:
 Accuracy: 87.1%,  loss: 0.299351

Test Error:
 Accuracy: 86.7%, loss: 0.349878

Epoch 24
----------------------------
Train Error:
 Accuracy: 87.1%,  loss: 0.296835

Test Error:
 Accuracy: 86.7%, loss: 0.350486

Epoch 25
----------------------------
Train Error:
 Accuracy: 87.2%,  loss: 0.294380

Test Error:
 Accuracy: 86.7%, loss: 0.351182

Epoch 26
----------------------------
Train Error:
 Accuracy: 87.2%,  loss: 0.291996

Test Error:
 Accuracy: 86.6%, loss: 0.351738

Epoch 27
----------------------------
Train Error:
 Accuracy: 87.3%,  loss: 0.289629

Test Error:
 Accuracy: 86.7%, loss: 0.352362

Epoch 28
----------------------------
Train Error:
 Accuracy: 87.4%,  loss: 0.287327

Test Error:
 Accuracy: 86.8%, loss: 0.353027

Epoch 29
----------------------------
Train Error:
 Accuracy: 87.4%,  loss: 0.285009

Test Error:
 Accuracy: 86.8%, loss: 0.354040

Epoch 30
----------------------------
Train Error:
 Accuracy: 87.5%,  loss: 0.282699

Test Error:
 Accuracy: 86.7%, loss: 0.354596

time:19.360673904418945
"""