from gensim.models import KeyedVectors
import re
import random

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

se_vectors = []
sy_vectors = []
flag = True

with open('questions-words.txt') as file:
    lines = file.readlines()
    for line in lines:
        line = line.replace('\n','').split(' ')
        if flag:
            if re.match(r'gram*', line[1]):
                flag = False
            else:
                se_vectors.append(line)
        else:
            sy_vectors.append(line)

random.seed(0)
se_vectors = random.sample(se_vectors, 100)
sy_vectors = random.sample(sy_vectors, 100)
se_score = 0
sy_score = 0


for vector1, vector2 in zip(se_vectors,sy_vectors):
    similar1 = model.most_similar(positive=[vector1[1],vector1[2]], negative=[vector1[0]], topn=1)
    similar2 = model.most_similar(positive=[vector2[1],vector2[2]], negative=[vector2[0]], topn=1)
    if vector1[3] == similar1[0][0]:
        se_score += 1
    if vector2[3] == similar2[0][0]:
        sy_score +=1

print('意味的アナロジー')
print(se_score/len(se_vectors))
print('文法的アナロジー')
print(sy_score/len(sy_vectors))





    

