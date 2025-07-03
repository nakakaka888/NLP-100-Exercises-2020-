import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn
from gensim.models import KeyedVectors
import numpy as np
import time

train_x = pd.read_csv('./section9/train.txt', sep='\t') 
train_y = pd.read_csv('./section9/train_y.txt', sep='\t') 
train_y = torch.tensor(train_y['CATEGORY'].values, dtype=torch.long)


test_x = pd.read_csv('./section9/test.txt', sep='\t') 
test_y = pd.read_csv('./section9/test_y.txt', sep='\t') 
test_y = torch.tensor(test_y['CATEGORY'].values, dtype=torch.long)

vectors = KeyedVectors.load_word2vec_format('./section7/GoogleNews-vectors-negative300.bin.gz', binary=True)

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


def my_collate_fn(batch):
    images, targets= list(zip(*batch))
    images = torch.stack(images)
    targets = torch.stack(targets)
    return images, targets

def emb_matrix(id_list, id_max):
    value_list = {}
    embedding_matrix = np.random.rand(id_max+1, 300)

    for i, (key, value) in enumerate(id_list.items()):

        if i > id_max:
            break

        if key in vectors:
            value_list[key] = value  
            embedding_matrix[value] = vectors[key]
    
    return torch.tensor(embedding_matrix, dtype=torch.float32)


class LSTM(nn.Module):
    def __init__(self, input_size, vocab_size, id_list, hidden_size=100, output_size=4):
        super().__init__()
        self.pre_matrix = emb_matrix(id_list, vocab_size)
        self.embedding = nn.Embedding.from_pretrained(self.pre_matrix, freeze=False, padding_idx = 0)
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=1) #batch_first=True in-output(batch_size, seq, feature_size)
        self.linear = nn.Linear(hidden_size, output_size)
        

    def forward(self, X, h=None):
        X = self.embedding(X) #in (batch_size, V) -> (batch_size, V, input_size)
        lstm_out, (hn, cn) = self.LSTM(X, h) #out (batch_size, V, hidden_size)
        output  = self.linear(lstm_out[:,-1,:])
        return output 

model = LSTM(input_size=300, vocab_size=id_max+1, id_list=id_list,  hidden_size=100, output_size=4)


learning_rate = 1e-2
batch_size = 16
epochs = 40


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


start = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    print("------Train Data------")
    cal_Accuracy(train_dataloader, model)
    print("------Test  Data------")
    cal_Accuracy(test_dataloader, model)
end = time.time()

print(f"time:{end - start}")

"""
Epoch 35
----------------------------
loss: 0.188024  [    0/10671]
loss: 0.507609  [ 1600/10671]
loss: 0.624804  [ 3200/10671]
loss: 0.357299  [ 4800/10671]
loss: 0.474902  [ 6400/10671]
loss: 0.269089  [ 8000/10671]
loss: 0.284999  [ 9600/10671]
------Train Data------
Accuracy: 85.1279%

------Test  Data------
Accuracy: 79.5352%

Epoch 36
----------------------------
loss: 0.221018  [    0/10671]
loss: 0.682161  [ 1600/10671]
loss: 0.319754  [ 3200/10671]
loss: 0.333746  [ 4800/10671]
loss: 0.186522  [ 6400/10671]
loss: 0.166532  [ 8000/10671]
loss: 0.377157  [ 9600/10671]
------Train Data------
Accuracy: 85.4559%

------Test  Data------
Accuracy: 78.3358%

Epoch 37
----------------------------
loss: 0.568716  [    0/10671]
loss: 0.394480  [ 1600/10671]
loss: 0.557574  [ 3200/10671]
loss: 0.152974  [ 4800/10671]
loss: 0.374017  [ 6400/10671]
loss: 0.449823  [ 8000/10671]
loss: 0.351597  [ 9600/10671]
------Train Data------
Accuracy: 87.0959%

------Test  Data------
Accuracy: 79.8351%

Epoch 38
----------------------------
loss: 0.311330  [    0/10671]
loss: 0.212184  [ 1600/10671]
loss: 0.499433  [ 3200/10671]
loss: 0.180101  [ 4800/10671]
loss: 0.680157  [ 6400/10671]
loss: 0.061626  [ 8000/10671]
loss: 0.139978  [ 9600/10671]
------Train Data------
Accuracy: 85.0155%

------Test  Data------
Accuracy: 79.5352%

Epoch 39
----------------------------
loss: 0.241454  [    0/10671]
loss: 0.212178  [ 1600/10671]
loss: 0.529618  [ 3200/10671]
loss: 0.455935  [ 4800/10671]
loss: 0.143924  [ 6400/10671]
loss: 0.202855  [ 8000/10671]
loss: 0.582402  [ 9600/10671]
------Train Data------
Accuracy: 88.0049%

------Test  Data------
Accuracy: 79.6102%

Epoch 40
----------------------------
loss: 0.184070  [    0/10671]
loss: 0.320356  [ 1600/10671]
loss: 0.152474  [ 3200/10671]
loss: 0.118685  [ 4800/10671]
loss: 0.464033  [ 6400/10671]
loss: 0.450850  [ 8000/10671]
loss: 0.180001  [ 9600/10671]
------Train Data------
Accuracy: 87.0022%

------Test  Data------
Accuracy: 79.3103%

time:244.8493320941925
"""


