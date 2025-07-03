import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn


train_x = pd.read_csv('./section9/train.txt', sep='\t') 
train_y = pd.read_csv('./section9/train_y.txt', sep='\t') 
train_y = torch.tensor(train_y['CATEGORY'].values, dtype=torch.long)


test_x = pd.read_csv('./section9/test.txt', sep='\t') 
test_y = pd.read_csv('./section9/test_y.txt', sep='\t') 
test_y = torch.tensor(test_y['CATEGORY'].values, dtype=torch.long)


def data_modify(data):
    data_x = data['TITLE']
    data_x = data_x.str.replace(r'\'s|[\'\"\:\.,\;\!\&\?\$]', '', regex=True)
    data_x = data_x.str.replace(r'\s-\s', ' ', regex=True)
    data_x = data_x.str.lower()

    sentence = [line.split() for line in data_x]
    
    return sentence

def word_count(sentence):
    word_list = {}
    for list in sentence:
        for word in list:
            if word not in word_list:
                word_list[word] = 1
            else:
                word_list[word] += 1

    word_list = sorted(word_list.items(), reverse=True, key = lambda x : x[1])

    rank_list = {}
    rank = 1
    for i, (item,key) in enumerate(word_list):

        if key < 2:
            rank_list[item] = 0
        else:
            rank_list[item] = rank
            rank += 1 
    return rank_list, rank-1
    
def get_id(words, rank_list):
    id_list = []
    for word in words:
        if  word in rank_list and rank_list[word] > 0:
            id_list.append(int(rank_list[word]) -1)
    return torch.tensor(id_list,  dtype=torch.long)

def one_hot_encode(list, id_list, id_max):
    tensor = torch.zeros(id_max)
    list_id = get_id(list, id_list)
    tensor[list_id] = 1
    return tensor


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx] , self.y[idx]


sentence_train = data_modify(train_x)
sentence_test  = data_modify(test_x)


#前処理
l = 0
id = 0
for i, list in enumerate(sentence_train):
    if len(list) > l:
        l = len(list)
        id = i
sentence_train.pop(id)
train_y = torch.cat((train_y[:id], train_y[id + 1:]))

id_list, id_max = word_count(sentence_train)


#padding
onehot_train = rnn.pad_sequence([get_id(list, id_list) for list in sentence_train], batch_first=True, padding_value=0)
onehot_test = rnn.pad_sequence([get_id(list, id_list) for list in sentence_test], batch_first=True, padding_value=0)

train_dataset = MyDataset(onehot_train, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = MyDataset(onehot_test, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

class LSTM(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size=100, output_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=1) #batch_first=True in-output(batch_size, seq, feature_size)
        self.linear = nn.Linear(hidden_size, output_size)
        

    def forward(self, X, h=None):
        X = self.embedding(X) #in (batch_size, V) -> (batch_size, V, input_size)
        lstm_out, (hn, cn) = self.LSTM(X, h) #out (batch_size, V, hidden_size)
        output  = self.linear(lstm_out[:,-1,:])
        return output 

model = LSTM(input_size=300, vocab_size=id_max+1, hidden_size=100, output_size=4)


learning_rate = 1e-2
batch_size = 16
epochs = 50


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

def train_loop(dataloader,  model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for  batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

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
    print(f"Accuracy: {(100*correct):>0.4f}%\n")



for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    print("------Train Data------")
    cal_Accuracy(train_dataloader, model)
    print("------Test  Data------")
    cal_Accuracy(test_dataloader, model)

"""
Epoch 40
----------------------------
loss: 0.971588  [    0/10671]
loss: 0.061959  [ 1600/10671]
loss: 0.179824  [ 3200/10671]
loss: 0.047282  [ 4800/10671]
loss: 0.011588  [ 6400/10671]
loss: 0.107387  [ 8000/10671]
loss: 0.027130  [ 9600/10671]
------Train Data------
Accuracy: 96.6264%

------Test  Data------
Accuracy: 77.2864%

Epoch 41
----------------------------
loss: 0.407282  [    0/10671]
loss: 0.073312  [ 1600/10671]
loss: 0.033959  [ 3200/10671]
loss: 0.064660  [ 4800/10671]
loss: 0.301957  [ 6400/10671]
loss: 0.291240  [ 8000/10671]
loss: 0.021984  [ 9600/10671]
------Train Data------
Accuracy: 97.1418%

------Test  Data------
Accuracy: 78.7856%

Epoch 42
----------------------------
loss: 0.224631  [    0/10671]
loss: 0.318924  [ 1600/10671]
loss: 0.020257  [ 3200/10671]
loss: 0.305074  [ 4800/10671]
loss: 0.075008  [ 6400/10671]
loss: 0.064068  [ 8000/10671]
loss: 0.296860  [ 9600/10671]
------Train Data------
Accuracy: 97.2917%

------Test  Data------
Accuracy: 78.0360%

Epoch 43
----------------------------
loss: 0.013342  [    0/10671]
loss: 0.050997  [ 1600/10671]
loss: 0.070332  [ 3200/10671]
loss: 0.015954  [ 4800/10671]
loss: 0.016964  [ 6400/10671]
loss: 0.330097  [ 8000/10671]
loss: 0.173302  [ 9600/10671]
------Train Data------
Accuracy: 97.0481%

------Test  Data------
Accuracy: 78.1109%

Epoch 44
----------------------------
loss: 0.619920  [    0/10671]
loss: 0.043815  [ 1600/10671]
loss: 0.338480  [ 3200/10671]
loss: 0.031342  [ 4800/10671]
loss: 0.022603  [ 6400/10671]
loss: 0.384216  [ 8000/10671]
loss: 0.045972  [ 9600/10671]
------Train Data------
Accuracy: 98.1164%

------Test  Data------
Accuracy: 78.9355%

Epoch 45
----------------------------
loss: 0.022770  [    0/10671]
loss: 0.033554  [ 1600/10671]
loss: 0.067414  [ 3200/10671]
loss: 0.019324  [ 4800/10671]
loss: 0.038176  [ 6400/10671]
loss: 0.032573  [ 8000/10671]
loss: 0.014863  [ 9600/10671]
------Train Data------
Accuracy: 98.1632%

------Test  Data------
Accuracy: 78.7106%

Epoch 46
----------------------------
loss: 0.014241  [    0/10671]
loss: 0.013936  [ 1600/10671]
loss: 0.024133  [ 3200/10671]
loss: 0.010260  [ 4800/10671]
loss: 0.221970  [ 6400/10671]
loss: 0.075442  [ 8000/10671]
loss: 0.045677  [ 9600/10671]
------Train Data------
Accuracy: 97.3292%

------Test  Data------
Accuracy: 79.2354%

Epoch 47
----------------------------
loss: 0.371775  [    0/10671]
loss: 0.210246  [ 1600/10671]
loss: 0.068686  [ 3200/10671]
loss: 0.024016  [ 4800/10671]
loss: 0.060481  [ 6400/10671]
loss: 0.027656  [ 8000/10671]
loss: 0.006723  [ 9600/10671]
------Train Data------
Accuracy: 97.7978%

------Test  Data------
Accuracy: 79.6102%

Epoch 48
----------------------------
loss: 0.041305  [    0/10671]
loss: 0.026418  [ 1600/10671]
loss: 0.012848  [ 3200/10671]
loss: 0.008181  [ 4800/10671]
loss: 0.053911  [ 6400/10671]
loss: 0.027259  [ 8000/10671]
loss: 0.031821  [ 9600/10671]
------Train Data------
Accuracy: 97.9290%

------Test  Data------
Accuracy: 78.4858%

Epoch 49
----------------------------
loss: 0.110204  [    0/10671]
loss: 0.459841  [ 1600/10671]
loss: 0.011684  [ 3200/10671]
loss: 0.014358  [ 4800/10671]
loss: 0.255636  [ 6400/10671]
loss: 0.027611  [ 8000/10671]
loss: 0.062981  [ 9600/10671]
------Train Data------
Accuracy: 98.3788%

------Test  Data------
Accuracy: 78.9355%

Epoch 50
----------------------------
loss: 0.072864  [    0/10671]
loss: 0.006802  [ 1600/10671]
loss: 0.076470  [ 3200/10671]
loss: 0.084342  [ 4800/10671]
loss: 0.050836  [ 6400/10671]
loss: 0.155023  [ 8000/10671]
loss: 0.049807  [ 9600/10671]
------Train Data------
Accuracy: 98.8942%

------Test  Data------
Accuracy: 79.0105%
"""