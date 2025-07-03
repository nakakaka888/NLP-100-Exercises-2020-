import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn


train_x = pd.read_csv('./section9/train.txt', sep='\t') 
train_y = pd.read_csv('./section9/train_y.txt', sep='\t') 
train_y = torch.tensor(train_y['CATEGORY'].values, dtype=torch.long)

def data_modify(data):
    data_x = data['TITLE']
    data_x = data_x.str.replace(r'\'s|[\'\"\:\.,\;\!\&\?\$]', '', regex=True)
    data_x = data_x.str.replace(r'\s-\s', ' ', regex=True)
    data_x = data_x.str.lower()

    sentence = [line.split() for line in data_x]
    
    return sentence

def word_count(sentence):
    word_list = {}
    max_len = 0
    for list in sentence:
        for word in list:
            max_len
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
        if rank_list[word] > 0:
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


sentence = data_modify(train_x)

#前処理
l = 0
id = 0
for i, list in enumerate(sentence):
    if len(list) > l:
        l = len(list)
        id = i
sentence.pop(id)
train_y = torch.cat((train_y[:id], train_y[id + 1:]))

id_list, id_max = word_count(sentence)

#padding
one_hot_tensor = rnn.pad_sequence([get_id(list, id_list) for list in sentence], batch_first=True)


class LSTM(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size=100, output_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300, padding_idx=0)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size) 
        self.linear = nn.Linear(hidden_size, output_size)
        

    def forward(self, X, h=None):
        X = self.embedding(X)
        lstm_out, _ = self.lstm(X, h)
        output  = self.linear(lstm_out)
        return output 

model = LSTM(input_size=300, vocab_size=id_max+1, hidden_size=50, output_size=4)
logits = model(one_hot_tensor)

proba = nn.Softmax(dim=-1)(logits)
print(proba)


"""
>実行結果
tensor([[[0.2270, 0.2791, 0.2416, 0.2523],
         [0.2009, 0.2530, 0.2707, 0.2753],
         [0.2498, 0.2482, 0.2542, 0.2478],
         ...,
         [0.2265, 0.2513, 0.2669, 0.2553],
         [0.2265, 0.2513, 0.2669, 0.2553],
         [0.2265, 0.2513, 0.2669, 0.2553]],

        [[0.2264, 0.2617, 0.2607, 0.2512],
         [0.2587, 0.2739, 0.2396, 0.2277],
         [0.2528, 0.2502, 0.2331, 0.2639],
         ...,
         [0.2229, 0.2505, 0.2684, 0.2582],
         [0.2229, 0.2505, 0.2684, 0.2582],
         [0.2229, 0.2505, 0.2684, 0.2582]],

        [[0.2097, 0.2688, 0.2275, 0.2940],
         [0.1642, 0.2379, 0.3217, 0.2762],
         [0.2223, 0.2572, 0.2763, 0.2443],
         ...,
         [0.2211, 0.2496, 0.2695, 0.2598],
         [0.2211, 0.2496, 0.2695, 0.2598],
         [0.2211, 0.2496, 0.2695, 0.2598]],

        ...,

        [[0.1958, 0.2434, 0.2876, 0.2733],
         [0.1874, 0.2258, 0.2758, 0.3110],
         [0.2670, 0.2389, 0.2682, 0.2260],
         ...,
         [0.2188, 0.2477, 0.2717, 0.2618],
         [0.2188, 0.2477, 0.2717, 0.2618],
         [0.2188, 0.2477, 0.2717, 0.2618]],

        [[0.2143, 0.2152, 0.3163, 0.2542],
         [0.2269, 0.2794, 0.2496, 0.2441],
         [0.2472, 0.2850, 0.2147, 0.2531],
         ...,
         [0.2188, 0.2477, 0.2717, 0.2618],
         [0.2188, 0.2477, 0.2717, 0.2618],
         [0.2188, 0.2477, 0.2717, 0.2618]],

        [[0.2475, 0.2410, 0.2850, 0.2264],
         [0.2192, 0.2412, 0.2629, 0.2767],
         [0.2166, 0.2936, 0.2270, 0.2629],
         ...,
         [0.2188, 0.2477, 0.2717, 0.2618],
         [0.2188, 0.2477, 0.2717, 0.2618],
         [0.2188, 0.2477, 0.2717, 0.2618]]], grad_fn=<SoftmaxBackward0>)
"""