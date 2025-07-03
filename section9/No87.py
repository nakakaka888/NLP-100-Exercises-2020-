import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
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


"""
CNN: 構成

単語埋め込みの次元数: dw = 300 emb後に次元
畳み込みのフィルターのサイズ: 3 トークン t-1 ,t,t+1
畳み込みのストライド: 1 トークン
畳み込みのパディング: あり
畳み込み演算後の各時刻のベクトルの次元数: dh = 50 隠れ層
畳み込み演算後に最大値プーリング（max pooling）を適用し,入力文をdh次元の隠れベクトルで表現

"""

class CNN(nn.Module):

    #入力(batch_size, 3, 28, 28) -> (batch_size=16 ,18, 300)

    def __init__(self, input_size, vocab_size, pooling_size, output_size=4):
        super().__init__()

        filter_size = 3
        self.embeding = nn.Embedding(vocab_size, input_size, padding_idx=0)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=pooling_size, kernel_size=(filter_size,input_size), stride=1, padding=(1,0))
        
        self.relu  = nn.ReLU()

        self.linear = nn.Linear(pooling_size, output_size)


    def forward(self, X):
        X = self.embeding(X) # (batch_size, seq, dim)
        X = X.unsqueeze(1)
        conv_out = self.conv1(X)  #out batch_size hidden_size
        conv_out = self.relu(conv_out).squeeze(3)
        pool_out = F.max_pool1d(conv_out, conv_out.size(2)) #max_pool1d(シーケンスデータ)はseqに対してpooling
        pool_out = pool_out.squeeze(2) 
        output   = self.linear(pool_out)
        return  output

model = CNN(input_size=300, vocab_size=id_max+1,pooling_size=50, output_size=4).to(device)

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3
batch_size = 16 
epochs = 15

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
Epoch 10
----------------------------
loss: 0.803006  [    0/10671]
loss: 0.566543  [ 1600/10671]
loss: 0.604335  [ 3200/10671]
loss: 0.452112  [ 4800/10671]
loss: 0.650982  [ 6400/10671]
loss: 0.660433  [ 8000/10671]
loss: 0.600517  [ 9600/10671]
------Train Data------
Accuracy: 77.4810%

------Test  Data------
Accuracy: 72.2639%

Epoch 11
----------------------------
loss: 0.698872  [    0/10671]
loss: 0.492333  [ 1600/10671]
loss: 0.799504  [ 3200/10671]
loss: 0.479347  [ 4800/10671]
loss: 0.925489  [ 6400/10671]
loss: 0.803641  [ 8000/10671]
loss: 0.558409  [ 9600/10671]
------Train Data------
Accuracy: 78.6524%

------Test  Data------
Accuracy: 72.6387%

Epoch 12
----------------------------
loss: 0.446468  [    0/10671]
loss: 0.498296  [ 1600/10671]
loss: 0.390380  [ 3200/10671]
loss: 0.391612  [ 4800/10671]
loss: 0.824917  [ 6400/10671]
loss: 0.669768  [ 8000/10671]
loss: 0.860449  [ 9600/10671]
------Train Data------
Accuracy: 80.3580%

------Test  Data------
Accuracy: 73.7631%

Epoch 13
----------------------------
loss: 0.458091  [    0/10671]
loss: 0.313537  [ 1600/10671]
loss: 0.654769  [ 3200/10671]
loss: 0.435160  [ 4800/10671]
loss: 0.586026  [ 6400/10671]
loss: 0.705202  [ 8000/10671]
loss: 0.607342  [ 9600/10671]
------Train Data------
Accuracy: 81.7262%

------Test  Data------
Accuracy: 74.3628%

Epoch 14
----------------------------
loss: 0.396693  [    0/10671]
loss: 0.738912  [ 1600/10671]
loss: 0.496786  [ 3200/10671]
loss: 0.409963  [ 4800/10671]
loss: 0.294427  [ 6400/10671]
loss: 0.433396  [ 8000/10671]
loss: 0.335432  [ 9600/10671]
------Train Data------
Accuracy: 83.1225%

------Test  Data------
Accuracy: 74.8876%

Epoch 15
----------------------------
loss: 0.349003  [    0/10671]
loss: 0.636609  [ 1600/10671]
loss: 0.326671  [ 3200/10671]
loss: 0.264183  [ 4800/10671]
loss: 0.460568  [ 6400/10671]
loss: 0.557246  [ 8000/10671]
loss: 0.648550  [ 9600/10671]
------Train Data------
Accuracy: 84.2470%

------Test  Data------
Accuracy: 75.8621%

time: 12.855918
"""