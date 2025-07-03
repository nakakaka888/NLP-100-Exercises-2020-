import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn
import time 

#データセット
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

#モデル
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size=50, output_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0)
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional=True) #batch_first=True in-output(batch_size, seq, feature_size)
        self.linear = nn.Linear(hidden_size*2, output_size)
        

    def forward(self, X, h=None):
        X = self.embedding(X) #in (batch_size, V) -> (batch_size, V, input_size)
        lstm_out, (hn, cn) = self.LSTM(X, h) #out (batch_size, V, hidden_size)
        output  = self.linear(lstm_out[:,-1,:])
        return output


#訓練
def train(dataloader,  model, device, loss_fn, optimizer):
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
def test(dataloader, model, device):
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad(): #勾配計算の無効
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    return 1 - correct


def get_optimizer(trial, model):

    learning_rate = trial.suggest_float('learing_rate',1e-6, 1e-3, log=True )

    optimizer_names = ["Adam", "AdamW", "RAdam"]
    optimizer_name = trial.suggest_categorical("optimizer", optimizer_names)

    if optimizer_name == optimizer_names[0]:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    elif optimizer_name == optimizer_names[1]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    else:
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)

    return optimizer


epoch = 5
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x = pd.read_csv('./section9/train.txt', sep='\t') 
    train_y = pd.read_csv('./section9/train_y.txt', sep='\t') 
    train_y = torch.tensor(train_y['CATEGORY'].values, dtype=torch.long)


    test_x = pd.read_csv('./section9/test.txt', sep='\t') 
    test_y = pd.read_csv('./section9/test_y.txt', sep='\t') 
    test_y = torch.tensor(test_y['CATEGORY'].values, dtype=torch.long)

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
    test_dataset  = MyDataset(onehot_test, test_y)

    #batchサイズ
    batch_sizes = trial.suggest_categorical("batchsize", [4, 8, 16, 32, 64])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_sizes)

    model = BidirectionalLSTM(input_size=300, vocab_size=id_max+1, hidden_size=50, output_size=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(trial, model)

    for step in range(epoch):
        train(train_dataloader, model, device, loss_fn, optimizer)
        correct = test(test_dataloader, model, device)
        
    return correct 

TRIAL_SIZE = 100
start = time.time()
study = optuna.create_study()
study.optimize(objective, n_trials=TRIAL_SIZE)

end =time.time()

print(study.best_params)

print(study.best_value)

print(f"time:{(end - start): 0.6f}")


"""
[I 2024-05-30 16:56:00,092] Trial 85 finished with value: 0.10944527736131937 and parameters: {'batchsize': 4, 'learing_rate': 0.0008727966031037671, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 16:56:37,189] Trial 86 finished with value: 0.12518740629685154 and parameters: {'batchsize': 4, 'learing_rate': 0.0008763457560170958, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 16:57:11,167] Trial 87 finished with value: 0.1356821589205397 and parameters: {'batchsize': 4, 'learing_rate': 0.0009229286036769906, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 16:57:43,771] Trial 88 finished with value: 0.12293853073463268 and parameters: {'batchsize': 4, 'learing_rate': 0.0009794299925193356, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 16:58:18,177] Trial 89 finished with value: 0.23688155922038978 and parameters: {'batchsize': 4, 'learing_rate': 5.088800981117908e-05, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 16:58:34,440] Trial 90 finished with value: 0.15892053973013498 and parameters: {'batchsize': 8, 'learing_rate': 0.0003992078487454988, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 16:59:07,011] Trial 91 finished with value: 0.13343328335832083 and parameters: {'batchsize': 4, 'learing_rate': 0.000989777211764763, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 16:59:39,107] Trial 92 finished with value: 0.13193403298350825 and parameters: {'batchsize': 4, 'learing_rate': 0.0005627472959377571, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 17:00:11,345] Trial 93 finished with value: 0.14392803598200898 and parameters: {'batchsize': 4, 'learing_rate': 0.0006659696073477918, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 17:00:43,622] Trial 94 finished with value: 0.13793103448275867 and parameters: {'batchsize': 4, 'learing_rate': 0.0008277069114570629, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 17:01:16,229] Trial 95 finished with value: 0.5689655172413793 and parameters: {'batchsize': 4, 'learing_rate': 4.568932071688721e-06, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 17:01:54,558] Trial 96 finished with value: 0.14842578710644683 and parameters: {'batchsize': 4, 'learing_rate': 0.00045548655374479455, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 17:01:57,278] Trial 97 finished with value: 0.1499250374812594 and parameters: {'batchsize': 64, 'learing_rate': 0.0009877285693767166, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 17:02:29,800] Trial 98 finished with value: 0.1214392803598201 and parameters: {'batchsize': 4, 'learing_rate': 0.0006806033757016307, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
[I 2024-05-30 17:02:34,269] Trial 99 finished with value: 0.17541229385307344 and parameters: {'batchsize': 32, 'learing_rate': 0.0007012655097798395, 'optimizer': 'Adam'}. Best is trial 85 with value: 0.10944527736131937.
time: 2428.940127
"""