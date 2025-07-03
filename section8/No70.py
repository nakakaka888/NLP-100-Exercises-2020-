import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
import torch
from tqdm import tqdm


"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

model = KeyedVectors.load_word2vec_format('./section7/GoogleNews-vectors-negative300.bin.gz', binary=True)

train = pd.read_csv('./section6/train.txt', sep='\t')
valid = pd.read_csv('./section6/valid.txt', sep='\t')
test  = pd.read_csv('./section6/test.txt', sep='\t')
categories = ['b', 't', 'e', 'm']

"""
ラベルの種類数 L = 4
正解ラベル
0 : ビジネス
1 : 科学技術
2 : エンターテインメント
3 : 健康
"""

#データの前処理
def data_const(df):
    data_x = df['TITLE']
    data_x = data_x.str.replace(r'\'s|[\'\"\:\.,\;\!\&\?\$]', '', regex=True)
    data_y = df['CATEGORY']
    data_y = [categories.index(label) for label in data_y]
    data_x = [line.split() for line in data_x]
    return data_x, data_y    


#単語をベクトルへ
def word_to_vector(data):
    np_array = np.array([model[word] for word in data if word in model])
    torch_data = torch.tensor(np_array)
    torch_avg = sum(torch_data) / len(torch_data)
    return torch_avg


train_x, train_y = data_const(train)
valid_x, valid_y = data_const(valid)
test_x , test_y  = data_const(test)



tensor_train = torch.stack([word_to_vector(line) for line in tqdm(train_x)] )
tensor_valid = torch.stack([word_to_vector(line) for line in tqdm(valid_x)])
tensor_test = torch.stack([word_to_vector(line) for line  in tqdm(test_x)])

print("tensor_train_size:", len(tensor_train))
print("tensor_valid_size:", len(tensor_valid))
print("tensor_test_size:", len(tensor_test))


train_y = pd.DataFrame({"CATEGORY":train_y})
train_y.to_csv('./section8/train_y.txt',index=False, sep='\t')

valid_y = pd.DataFrame({"CATEGORY":valid_y})
valid_y.to_csv('./section8/valid_y.txt',index=False, sep='\t')

test_y  = pd.DataFrame({"CATEGORY":test_y})
test_y.to_csv('./section8/test_y.txt',index=False, sep='\t')


torch.save(tensor_train, './section8/tensor_train.pth')
torch.save(tensor_valid, './section8/tensor_valid.pth')
torch.save(tensor_test, './section8/tensor_test.pth')

"""
>出力結果
tensor_train_size: 10672
tensor_valid_size: 1334
tensor_test_size: 1334
"""