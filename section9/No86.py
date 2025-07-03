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
onehot_train = rnn.pad_sequence([get_id(list, id_list) for list in sentence_train], batch_first=True, padding_value=0)
onehot_test = rnn.pad_sequence([get_id(list, id_list) for list in sentence_test], batch_first=True, padding_value=0)

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

model = CNN(input_size=300, vocab_size=id_max+1,pooling_size=50, output_size=4)

for i, (X,y) in enumerate(train_dataloader):
    preba = model(X)
    proba = nn.Softmax(dim=-1)(preba)
    print(proba)
    break

"""
tensor([[0.2421, 0.1388, 0.2644, 0.3547],
        [0.3165, 0.0982, 0.3119, 0.2734],
        [0.2268, 0.1417, 0.2828, 0.3487],
        [0.2611, 0.1516, 0.2725, 0.3147],
        [0.2802, 0.1350, 0.2833, 0.3015],
        [0.1835, 0.1261, 0.3517, 0.3387],
        [0.3179, 0.1041, 0.2557, 0.3222],
        [0.2196, 0.1094, 0.3650, 0.3061],
        [0.2368, 0.1439, 0.2285, 0.3909],
        [0.2594, 0.1034, 0.3122, 0.3250],
        [0.1967, 0.1425, 0.3088, 0.3520],
        [0.2534, 0.0899, 0.3142, 0.3424],
        [0.3475, 0.0956, 0.2679, 0.2891],
        [0.2385, 0.1166, 0.3436, 0.3013],
        [0.3401, 0.1468, 0.2703, 0.2428],
        [0.2373, 0.1709, 0.3109, 0.2808]], grad_fn=<SoftmaxBackward0>)
"""