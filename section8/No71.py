import torch
import torch.nn as nn


"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

train = torch.load('./section8/tensor_train.pth')


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(300, 4)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        logits = self.linear(x)
        return logits


model = NeuralNetwork()

pred_prob_train = torch.stack([nn.Softmax(dim=-1)(model(data)) for data in train])

print("==train shape==\n",pred_prob_train)

"""
>出力結果
==train shape==
 tensor([[0.3033, 0.2106, 0.2452, 0.2409],
        [0.2837, 0.2427, 0.2772, 0.1964],
        [0.2531, 0.2589, 0.2593, 0.2287],
        ...,
        [0.3547, 0.1965, 0.2106, 0.2383],
        [0.2942, 0.2076, 0.2231, 0.2751],
        [0.2361, 0.2187, 0.2966, 0.2485]], grad_fn=<StackBackward0>)
"""