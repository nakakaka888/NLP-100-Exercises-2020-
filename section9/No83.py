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
print(device)

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


def my_collate_fn(batch):
    images, targets= list(zip(*batch))
    images = torch.stack(images)
    targets = torch.stack(targets)
    return images, targets

class LSTM(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size=100, output_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0)
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True) #batch_first=True in-output(batch_size, seq, feature_size)
        self.linear = nn.Linear(hidden_size, output_size)
        

    def forward(self, X, h=None):
        X = self.embedding(X) #in (batch_size, V) -> (batch_size, V, input_size)
        lstm_out, (hn, cn) = self.LSTM(X, h) #out (batch_size, V, hidden_size)
        output  = self.linear(lstm_out[:,-1,:])
        return output 

model = LSTM(input_size=300, vocab_size=id_max+1, hidden_size=50, output_size=4).to(device)

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

print(f"time:{end - start}")

"""
>実行結果
Epoch 1
----------------------------
loss: 1.404051  [    0/10671]
loss: 1.340979  [ 1600/10671]
loss: 1.271354  [ 3200/10671]
loss: 1.201820  [ 4800/10671]
loss: 1.190849  [ 6400/10671]
loss: 1.176098  [ 8000/10671]
loss: 1.115577  [ 9600/10671]
------Train Data------
Accuracy: 42.0298%

------Test  Data------
Accuracy: 42.7286%

Epoch 2
----------------------------
loss: 1.247235  [    0/10671]
loss: 1.166746  [ 1600/10671]
loss: 1.146369  [ 3200/10671]
loss: 1.060397  [ 4800/10671]
loss: 1.206901  [ 6400/10671]
loss: 1.472846  [ 8000/10671]
loss: 1.295510  [ 9600/10671]
------Train Data------
Accuracy: 42.0485%

------Test  Data------
Accuracy: 42.8036%

Epoch 3
----------------------------
loss: 1.103839  [    0/10671]
loss: 1.075804  [ 1600/10671]
loss: 1.084201  [ 3200/10671]
loss: 1.242195  [ 4800/10671]
loss: 0.977130  [ 6400/10671]
loss: 1.294887  [ 8000/10671]
loss: 1.401681  [ 9600/10671]
------Train Data------
Accuracy: 42.0017%

------Test  Data------
Accuracy: 42.8036%

Epoch 4
----------------------------
loss: 1.310668  [    0/10671]
loss: 1.221232  [ 1600/10671]
loss: 1.181862  [ 3200/10671]
loss: 1.084683  [ 4800/10671]
loss: 1.164169  [ 6400/10671]
loss: 1.389779  [ 8000/10671]
loss: 1.308316  [ 9600/10671]
------Train Data------
Accuracy: 42.1329%

------Test  Data------
Accuracy: 43.1034%

Epoch 5
----------------------------
loss: 1.014719  [    0/10671]
loss: 1.234567  [ 1600/10671]
loss: 1.082958  [ 3200/10671]
loss: 1.062092  [ 4800/10671]
loss: 1.243454  [ 6400/10671]
loss: 1.086878  [ 8000/10671]
loss: 1.246664  [ 9600/10671]
------Train Data------
Accuracy: 42.1048%

------Test  Data------
Accuracy: 43.0285%

Epoch 6
----------------------------
loss: 1.192700  [    0/10671]
loss: 1.432974  [ 1600/10671]
loss: 1.091139  [ 3200/10671]
loss: 0.990518  [ 4800/10671]
loss: 1.088880  [ 6400/10671]
loss: 1.274205  [ 8000/10671]
loss: 1.073368  [ 9600/10671]
------Train Data------
Accuracy: 42.4328%

------Test  Data------
Accuracy: 43.6282%

Epoch 7
----------------------------
loss: 1.203211  [    0/10671]
loss: 1.169215  [ 1600/10671]
loss: 1.120302  [ 3200/10671]
loss: 1.247727  [ 4800/10671]
loss: 1.141284  [ 6400/10671]
loss: 1.082922  [ 8000/10671]
loss: 1.422305  [ 9600/10671]
------Train Data------
Accuracy: 43.8103%

------Test  Data------
Accuracy: 45.0525%

Epoch 8
----------------------------
loss: 1.085019  [    0/10671]
loss: 1.052745  [ 1600/10671]
loss: 0.971303  [ 3200/10671]
loss: 1.079916  [ 4800/10671]
loss: 1.246620  [ 6400/10671]
loss: 1.359320  [ 8000/10671]
loss: 0.972600  [ 9600/10671]
------Train Data------
Accuracy: 42.4890%

------Test  Data------
Accuracy: 43.5532%

Epoch 9
----------------------------
loss: 1.296797  [    0/10671]
loss: 1.163162  [ 1600/10671]
loss: 1.278621  [ 3200/10671]
loss: 1.077041  [ 4800/10671]
loss: 1.175497  [ 6400/10671]
loss: 0.968087  [ 8000/10671]
loss: 1.388887  [ 9600/10671]
------Train Data------
Accuracy: 43.0138%

------Test  Data------
Accuracy: 44.3028%

Epoch 10
----------------------------
loss: 1.024812  [    0/10671]
loss: 1.246284  [ 1600/10671]
loss: 1.196996  [ 3200/10671]
loss: 1.202856  [ 4800/10671]
loss: 1.045426  [ 6400/10671]
loss: 1.389706  [ 8000/10671]
loss: 0.968412  [ 9600/10671]
------Train Data------
Accuracy: 42.2266%

------Test  Data------
Accuracy: 43.0285%

Epoch 11
----------------------------
loss: 0.876194  [    0/10671]
loss: 1.160333  [ 1600/10671]
loss: 1.251396  [ 3200/10671]
loss: 0.971405  [ 4800/10671]
loss: 1.280012  [ 6400/10671]
loss: 1.110947  [ 8000/10671]
loss: 1.196504  [ 9600/10671]
------Train Data------
Accuracy: 42.9201%

------Test  Data------
Accuracy: 44.1529%

Epoch 12
----------------------------
loss: 1.237766  [    0/10671]
loss: 1.131872  [ 1600/10671]
loss: 1.001026  [ 3200/10671]
loss: 1.096627  [ 4800/10671]
loss: 1.315286  [ 6400/10671]
loss: 1.081899  [ 8000/10671]
loss: 1.199160  [ 9600/10671]
------Train Data------
Accuracy: 44.2133%

------Test  Data------
Accuracy: 46.0270%

Epoch 13
----------------------------
loss: 1.172246  [    0/10671]
loss: 1.267295  [ 1600/10671]
loss: 0.977531  [ 3200/10671]
loss: 1.536878  [ 4800/10671]
loss: 1.339694  [ 6400/10671]
loss: 1.295365  [ 8000/10671]
loss: 1.130854  [ 9600/10671]
------Train Data------
Accuracy: 44.4851%

------Test  Data------
Accuracy: 46.7016%

Epoch 14
----------------------------
loss: 1.366189  [    0/10671]
loss: 1.011184  [ 1600/10671]
loss: 1.007408  [ 3200/10671]
loss: 1.058576  [ 4800/10671]
loss: 1.264380  [ 6400/10671]
loss: 1.248440  [ 8000/10671]
loss: 1.175165  [ 9600/10671]
------Train Data------
Accuracy: 45.3941%

------Test  Data------
Accuracy: 47.4513%

Epoch 15
----------------------------
loss: 1.346824  [    0/10671]
loss: 1.128950  [ 1600/10671]
loss: 1.100553  [ 3200/10671]
loss: 1.060507  [ 4800/10671]
loss: 1.052877  [ 6400/10671]
loss: 0.883896  [ 8000/10671]
loss: 1.065078  [ 9600/10671]
------Train Data------
Accuracy: 45.4878%

------Test  Data------
Accuracy: 47.9010%

Epoch 16
----------------------------
loss: 1.264194  [    0/10671]
loss: 0.983700  [ 1600/10671]
loss: 1.348349  [ 3200/10671]
loss: 1.429294  [ 4800/10671]
loss: 0.989502  [ 6400/10671]
loss: 1.089160  [ 8000/10671]
loss: 1.193165  [ 9600/10671]
------Train Data------
Accuracy: 49.5549%

------Test  Data------
Accuracy: 52.2489%

Epoch 17
----------------------------
loss: 1.152058  [    0/10671]
loss: 1.417416  [ 1600/10671]
loss: 1.158285  [ 3200/10671]
loss: 1.073565  [ 4800/10671]
loss: 1.222244  [ 6400/10671]
loss: 1.221043  [ 8000/10671]
loss: 1.325037  [ 9600/10671]
------Train Data------
Accuracy: 49.7517%

------Test  Data------
Accuracy: 52.6237%

Epoch 18
----------------------------
loss: 1.076989  [    0/10671]
loss: 1.372444  [ 1600/10671]
loss: 0.854351  [ 3200/10671]
loss: 1.413174  [ 4800/10671]
loss: 1.168833  [ 6400/10671]
loss: 1.079256  [ 8000/10671]
loss: 1.272824  [ 9600/10671]
------Train Data------
Accuracy: 52.3662%

------Test  Data------
Accuracy: 53.8231%

Epoch 19
----------------------------
loss: 1.071545  [    0/10671]
loss: 0.954281  [ 1600/10671]
loss: 0.931191  [ 3200/10671]
loss: 0.945641  [ 4800/10671]
loss: 1.354577  [ 6400/10671]
loss: 1.290707  [ 8000/10671]
loss: 0.932629  [ 9600/10671]
------Train Data------
Accuracy: 54.8683%

------Test  Data------
Accuracy: 55.9970%

Epoch 20
----------------------------
loss: 1.042193  [    0/10671]
loss: 0.961169  [ 1600/10671]
loss: 1.100306  [ 3200/10671]
loss: 1.098840  [ 4800/10671]
loss: 1.151915  [ 6400/10671]
loss: 1.199385  [ 8000/10671]
loss: 0.959152  [ 9600/10671]
------Train Data------
Accuracy: 62.9182%

------Test  Data------
Accuracy: 60.7196%

Epoch 21
----------------------------
loss: 1.011038  [    0/10671]
loss: 1.112194  [ 1600/10671]
loss: 0.919744  [ 3200/10671]
loss: 0.982871  [ 4800/10671]
loss: 0.763048  [ 6400/10671]
loss: 0.736170  [ 8000/10671]
loss: 1.207825  [ 9600/10671]
------Train Data------
Accuracy: 70.5370%

------Test  Data------
Accuracy: 66.0420%

Epoch 22
----------------------------
loss: 0.912625  [    0/10671]
loss: 0.548258  [ 1600/10671]
loss: 1.116148  [ 3200/10671]
loss: 1.276002  [ 4800/10671]
loss: 0.930053  [ 6400/10671]
loss: 0.936360  [ 8000/10671]
loss: 1.003132  [ 9600/10671]
------Train Data------
Accuracy: 74.4448%

------Test  Data------
Accuracy: 70.5397%

Epoch 23
----------------------------
loss: 0.837644  [    0/10671]
loss: 0.639128  [ 1600/10671]
loss: 0.799509  [ 3200/10671]
loss: 0.728795  [ 4800/10671]
loss: 0.699061  [ 6400/10671]
loss: 0.526788  [ 8000/10671]
loss: 0.661469  [ 9600/10671]
------Train Data------
Accuracy: 77.1343%

------Test  Data------
Accuracy: 72.5637%

Epoch 24
----------------------------
loss: 0.760039  [    0/10671]
loss: 0.788286  [ 1600/10671]
loss: 1.001634  [ 3200/10671]
loss: 0.446651  [ 4800/10671]
loss: 0.824242  [ 6400/10671]
loss: 0.729467  [ 8000/10671]
loss: 0.801703  [ 9600/10671]
------Train Data------
Accuracy: 77.8371%

------Test  Data------
Accuracy: 72.7136%

Epoch 25
----------------------------
loss: 0.827503  [    0/10671]
loss: 0.426700  [ 1600/10671]
loss: 0.569665  [ 3200/10671]
loss: 1.117093  [ 4800/10671]
loss: 0.655127  [ 6400/10671]
loss: 0.963704  [ 8000/10671]
loss: 0.272638  [ 9600/10671]
------Train Data------
Accuracy: 79.6551%

------Test  Data------
Accuracy: 74.1379%

Epoch 26
----------------------------
loss: 0.541425  [    0/10671]
loss: 0.543397  [ 1600/10671]
loss: 0.593620  [ 3200/10671]
loss: 0.439462  [ 4800/10671]
loss: 0.576289  [ 6400/10671]
loss: 0.581943  [ 8000/10671]
loss: 0.116094  [ 9600/10671]
------Train Data------
Accuracy: 81.7543%

------Test  Data------
Accuracy: 74.8126%

Epoch 27
----------------------------
loss: 0.773848  [    0/10671]
loss: 0.531538  [ 1600/10671]
loss: 0.707725  [ 3200/10671]
loss: 0.500937  [ 4800/10671]
loss: 0.245671  [ 6400/10671]
loss: 0.834736  [ 8000/10671]
loss: 0.771354  [ 9600/10671]
------Train Data------
Accuracy: 83.3568%

------Test  Data------
Accuracy: 75.2624%

Epoch 28
----------------------------
loss: 0.545640  [    0/10671]
loss: 0.541060  [ 1600/10671]
loss: 0.394212  [ 3200/10671]
loss: 0.199082  [ 4800/10671]
loss: 0.327565  [ 6400/10671]
loss: 0.511966  [ 8000/10671]
loss: 0.770156  [ 9600/10671]
------Train Data------
Accuracy: 85.1185%

------Test  Data------
Accuracy: 74.6627%

Epoch 29
----------------------------
loss: 0.533635  [    0/10671]
loss: 0.537774  [ 1600/10671]
loss: 0.237897  [ 3200/10671]
loss: 0.477791  [ 4800/10671]
loss: 0.558545  [ 6400/10671]
loss: 0.368436  [ 8000/10671]
loss: 0.470530  [ 9600/10671]
------Train Data------
Accuracy: 84.0221%

------Test  Data------
Accuracy: 73.0135%

Epoch 30
----------------------------
loss: 0.406611  [    0/10671]
loss: 0.505022  [ 1600/10671]
loss: 0.346884  [ 3200/10671]
loss: 0.569732  [ 4800/10671]
loss: 0.389204  [ 6400/10671]
loss: 0.530591  [ 8000/10671]
loss: 0.201271  [ 9600/10671]
------Train Data------
Accuracy: 86.4774%

------Test  Data------
Accuracy: 75.1124%

time:28.07912039756775
"""