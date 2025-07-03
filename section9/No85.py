import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn
import time 

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_x = pd.read_csv('./section9/train.txt', sep='\t') 
train_y = pd.read_csv('./section9/train_y.txt', sep='\t') 
train_y = torch.tensor(train_y['CATEGORY'].values, dtype=torch.long).to(device)


test_x = pd.read_csv('./section9/test.txt', sep='\t') 
test_y = pd.read_csv('./section9/test_y.txt', sep='\t') 
test_y = torch.tensor(test_y['CATEGORY'].values, dtype=torch.long).to(device)

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
onehot_train = rnn.pad_sequence([get_id(list, id_list) for list in sentence_train], batch_first=True, padding_value=0).to(device)

onehot_test = rnn.pad_sequence([get_id(list, id_list) for list in sentence_test], batch_first=True, padding_value=0).to(device)


train_dataset = MyDataset(onehot_train, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = MyDataset(onehot_test, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size=50, output_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional=True) #batch_first=True in-output(batch_size, seq, feature_size)
        self.linear = nn.Linear(hidden_size*2, output_size)
        

    def forward(self, X, h=None):
        X = self.embedding(X) #in (batch_size, V) -> (batch_size, V, input_size)
        lstm_out, (hn, cn) = self.LSTM(X, h) #out (batch_size, V, hidden_size)
        lstm_out = torch.mean(lstm_out, dim=1)
        output  = self.linear(lstm_out)
        return output

model = BidirectionalLSTM(input_size=300, vocab_size=id_max+1, hidden_size=100, output_size=4).to(device)


learning_rate = 1e-2
batch_size = 16
epochs = 30


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

print(f"time:{(end - start): 0.6f}")


"""
Epoch 20
----------------------------
loss: 0.458503  [    0/10671]
loss: 0.345747  [ 1600/10671]
loss: 0.379522  [ 3200/10671]
loss: 0.313660  [ 4800/10671]
loss: 0.600936  [ 6400/10671]
loss: 0.343385  [ 8000/10671]
loss: 0.102843  [ 9600/10671]
------Train Data------
Accuracy: 84.6219%

------Test  Data------
Accuracy: 77.2114%

Epoch 21
----------------------------
loss: 0.158169  [    0/10671]
loss: 0.442232  [ 1600/10671]
loss: 0.143788  [ 3200/10671]
loss: 0.552449  [ 4800/10671]
loss: 0.349524  [ 6400/10671]
loss: 0.160518  [ 8000/10671]
loss: 0.298241  [ 9600/10671]
------Train Data------
Accuracy: 83.7691%

------Test  Data------
Accuracy: 76.6117%

Epoch 22
----------------------------
loss: 0.059780  [    0/10671]
loss: 0.489756  [ 1600/10671]
loss: 0.249933  [ 3200/10671]
loss: 0.403543  [ 4800/10671]
loss: 0.156770  [ 6400/10671]
loss: 0.345596  [ 8000/10671]
loss: 0.628328  [ 9600/10671]
------Train Data------
Accuracy: 90.0103%

------Test  Data------
Accuracy: 78.5607%

Epoch 23
----------------------------
loss: 0.063740  [    0/10671]
loss: 0.184947  [ 1600/10671]
loss: 0.100358  [ 3200/10671]
loss: 0.233225  [ 4800/10671]
loss: 0.340389  [ 6400/10671]
loss: 1.025688  [ 8000/10671]
loss: 0.292320  [ 9600/10671]
------Train Data------
Accuracy: 88.7733%

------Test  Data------
Accuracy: 77.5862%

Epoch 24
----------------------------
loss: 0.612340  [    0/10671]
loss: 0.278589  [ 1600/10671]
loss: 0.078965  [ 3200/10671]
loss: 0.269737  [ 4800/10671]
loss: 0.371377  [ 6400/10671]
loss: 0.053714  [ 8000/10671]
loss: 0.177245  [ 9600/10671]
------Train Data------
Accuracy: 90.6194%

------Test  Data------
Accuracy: 78.0360%

Epoch 25
----------------------------
loss: 0.180683  [    0/10671]
loss: 0.049172  [ 1600/10671]
loss: 0.148032  [ 3200/10671]
loss: 0.377621  [ 4800/10671]
loss: 0.093584  [ 6400/10671]
loss: 0.217534  [ 8000/10671]
loss: 0.243108  [ 9600/10671]
------Train Data------
Accuracy: 86.9178%

------Test  Data------
Accuracy: 76.0120%

Epoch 26
----------------------------
loss: 0.206621  [    0/10671]
loss: 0.197227  [ 1600/10671]
loss: 0.117855  [ 3200/10671]
loss: 0.162320  [ 4800/10671]
loss: 0.089778  [ 6400/10671]
loss: 0.149491  [ 8000/10671]
loss: 0.113090  [ 9600/10671]
------Train Data------
Accuracy: 92.2219%

------Test  Data------
Accuracy: 78.3358%

Epoch 27
----------------------------
loss: 0.095921  [    0/10671]
loss: 0.187014  [ 1600/10671]
loss: 0.276913  [ 3200/10671]
loss: 0.478874  [ 4800/10671]
loss: 0.181161  [ 6400/10671]
loss: 0.203211  [ 8000/10671]
loss: 0.147759  [ 9600/10671]
------Train Data------
Accuracy: 93.0934%

------Test  Data------
Accuracy: 78.9355%

Epoch 28
----------------------------
loss: 0.143213  [    0/10671]
loss: 0.165428  [ 1600/10671]
loss: 0.407077  [ 3200/10671]
loss: 0.101284  [ 4800/10671]
loss: 0.181606  [ 6400/10671]
loss: 0.544565  [ 8000/10671]
loss: 0.254405  [ 9600/10671]
------Train Data------
Accuracy: 92.3718%

------Test  Data------
Accuracy: 77.8111%

Epoch 29
----------------------------
loss: 0.010996  [    0/10671]
loss: 0.406305  [ 1600/10671]
loss: 0.199849  [ 3200/10671]
loss: 0.324718  [ 4800/10671]
loss: 0.221836  [ 6400/10671]
loss: 0.194419  [ 8000/10671]
loss: 0.134753  [ 9600/10671]
------Train Data------
Accuracy: 93.4402%

------Test  Data------
Accuracy: 79.3103%

Epoch 30
----------------------------
loss: 0.344747  [    0/10671]
loss: 0.117276  [ 1600/10671]
loss: 0.053652  [ 3200/10671]
loss: 0.293304  [ 4800/10671]
loss: 0.177281  [ 6400/10671]
loss: 0.059613  [ 8000/10671]
loss: 0.067985  [ 9600/10671]
------Train Data------
Accuracy: 95.1551%

------Test  Data------
Accuracy: 78.3358%

time: 54.187794
"""