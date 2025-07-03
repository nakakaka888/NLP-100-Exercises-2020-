import torch
import torch.nn as nn
import pandas as pd
import numpy as np


"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

train_x = torch.load('./section8/tensor_train.pth')
train_y = pd.read_csv('./section8/train_y.txt')

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(300, 4)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        logits = self.linear(x)
        return logits


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()

t = torch.tensor(train_y['CATEGORY'].values)
pred_prob_train = torch.stack([nn.Softmax(dim=-1)(model(data)) for data in train_x])
loss = loss_fn(pred_prob_train, t)

model.zero_grad()
loss.backward()

print('損失:', loss)
print('勾配:', model.linear.weight.grad)

"""
>出力結果
損失: tensor(1.3881, grad_fn=<NllLossBackward0>)
勾配: tensor([[ 0.0005, -0.0013,  0.0009,  ..., -0.0022, -0.0031,  0.0035],
             [ 0.0007,  0.0011, -0.0018,  ..., -0.0001,  0.0024, -0.0007],
             [-0.0026, -0.0011,  0.0029,  ...,  0.0028, -0.0012, -0.0010],
             [ 0.0014,  0.0013, -0.0021,  ..., -0.0005,  0.0019, -0.0017]])
"""