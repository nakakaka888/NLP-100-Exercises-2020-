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

class TextCNN(nn.Module):
    def __init__(self, vocab_size, dw, dh, num_classes, embedding_matrix=None):
        super(TextCNN, self).__init__()
        # 単語埋め込み層
        self.embedding = nn.Embedding(vocab_size, dw)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False  # 埋め込みを固定する場合
        
        # 畳み込み層
        self.conv = nn.Conv2d(1, dh, (3, dw), stride=1, padding=(1, 0))
        
        # 線形変換層
        self.fc = nn.Linear(dh, num_classes)
        
    def forward(self, x):
        # 埋め込み層
        x = self.embedding(x).unsqueeze(1)  # (batch_size, 1, sequence_length, dw)
        # 畳み込み層 + ReLU
        x = F.relu(self.conv(x)).squeeze(3)  # (batch_size, dh, sequence_length)
        print(x.shape)
        # 最大値プーリング層
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # (batch_size, dh)
        
        # 線形変換層
        x = self.fc(x)  # (batch_size, num_classes)
        return x


model = TextCNN(vocab_size=id_max+1,dw=300, dh=50, num_classes=4)

for i, (X,y) in enumerate(train_dataloader):
    preb = model(X)
    print(preb.shape)
    break