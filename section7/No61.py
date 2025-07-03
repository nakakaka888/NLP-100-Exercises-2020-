from gensim.models import KeyedVectors
import numpy as np

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

model = KeyedVectors.load_word2vec_format('./section7/GoogleNews-vectors-negative300.bin.gz', binary=True)

united = model['United_States']
us     = model['US']

cos_s = np.dot(united, us) / (np.linalg.norm(united) * np.linalg.norm(us))
print(cos_s)

"""
>実行結果
0.45223224
"""