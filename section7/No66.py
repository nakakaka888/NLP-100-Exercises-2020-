from gensim.models import KeyedVectors
import pandas as pd
from scipy.stats import pearsonr

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def cal_sim(df):
    word1 = df['Word 1']
    word2 = df['Word 2']
    return model.similarity(word1,word2)

df = pd.read_csv('combined.csv', sep=',')

df['Similarity'] = df.apply(cal_sim,axis=1)


print(spearman(df['Similarity'], df['Human (mean)']))

"""
>実行結果
PearsonRResult(statistic=0.6525349618875618, pvalue=3.373418454306436e-44)
"""
