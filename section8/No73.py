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
train_dataset = MyDataset(train_x,train_y)
train_dataloader = DataLoader(train_dataset, batch_size=8)


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3
batch_size = 8
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


for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
print("DONE")

"""
>出力結果
Epoch 5
----------------------------
loss: 0.875069  [    0/10672]
loss: 0.937282  [  800/10672]
loss: 0.999556  [ 1600/10672]
loss: 0.741470  [ 2400/10672]
loss: 0.725758  [ 3200/10672]
loss: 0.913637  [ 4000/10672]
loss: 0.906681  [ 4800/10672]
loss: 0.755660  [ 5600/10672]
loss: 0.789099  [ 6400/10672]
loss: 0.726070  [ 7200/10672]
loss: 0.656878  [ 8000/10672]
loss: 0.710205  [ 8800/10672]
loss: 1.295896  [ 9600/10672]
loss: 0.662868  [10400/10672]
Epoch 6
----------------------------
loss: 0.834778  [    0/10672]
loss: 0.897308  [  800/10672]
loss: 0.974946  [ 1600/10672]
loss: 0.692908  [ 2400/10672]
loss: 0.674765  [ 3200/10672]
loss: 0.879347  [ 4000/10672]
loss: 0.868786  [ 4800/10672]
loss: 0.709711  [ 5600/10672]
loss: 0.747728  [ 6400/10672]
loss: 0.685094  [ 7200/10672]
loss: 0.608485  [ 8000/10672]
loss: 0.669394  [ 8800/10672]
loss: 1.283835  [ 9600/10672]
loss: 0.616941  [10400/10672]
Epoch 7
----------------------------
loss: 0.801361  [    0/10672]
loss: 0.864418  [  800/10672]
loss: 0.954612  [ 1600/10672]
loss: 0.654148  [ 2400/10672]
loss: 0.633967  [ 3200/10672]
loss: 0.850760  [ 4000/10672]
loss: 0.837589  [ 4800/10672]
loss: 0.672331  [ 5600/10672]
loss: 0.713741  [ 6400/10672]
loss: 0.651515  [ 7200/10672]
loss: 0.568580  [ 8000/10672]
loss: 0.635810  [ 8800/10672]
loss: 1.270566  [ 9600/10672]
loss: 0.579083  [10400/10672]
Epoch 8
----------------------------
loss: 0.772855  [    0/10672]
loss: 0.836494  [  800/10672]
loss: 0.937214  [ 1600/10672]
loss: 0.622268  [ 2400/10672]
loss: 0.600413  [ 3200/10672]
loss: 0.826112  [ 4000/10672]
loss: 0.811140  [ 4800/10672]
loss: 0.641141  [ 5600/10672]
loss: 0.685209  [ 6400/10672]
loss: 0.623277  [ 7200/10672]
loss: 0.534979  [ 8000/10672]
loss: 0.607548  [ 8800/10672]
loss: 1.256714  [ 9600/10672]
loss: 0.547252  [10400/10672]
Epoch 9
----------------------------
loss: 0.748054  [    0/10672]
loss: 0.812249  [  800/10672]
loss: 0.921964  [ 1600/10672]
loss: 0.595457  [ 2400/10672]
loss: 0.572262  [ 3200/10672]
loss: 0.804377  [ 4000/10672]
loss: 0.788262  [ 4800/10672]
loss: 0.614598  [ 5600/10672]
loss: 0.660820  [ 6400/10672]
loss: 0.599062  [ 7200/10672]
loss: 0.506207  [ 8000/10672]
loss: 0.583336  [ 8800/10672]
loss: 1.242658  [ 9600/10672]
loss: 0.520049  [10400/10672]
Epoch 10
----------------------------
loss: 0.726160  [    0/10672]
loss: 0.790839  [  800/10672]
loss: 0.908360  [ 1600/10672]
loss: 0.572511  [ 2400/10672]
loss: 0.548281  [ 3200/10672]
loss: 0.784900  [ 4000/10672]
loss: 0.768185  [ 4800/10672]
loss: 0.591646  [ 5600/10672]
loss: 0.639656  [ 6400/10672]
loss: 0.577971  [ 7200/10672]
loss: 0.481222  [ 8000/10672]
loss: 0.562284  [ 8800/10672]
loss: 1.228633  [ 9600/10672]
loss: 0.496481  [10400/10672]
DONE
"""