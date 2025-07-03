from gensim.models import KeyedVectors
import re
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use('Agg') # -----(1)

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
countries = []

with open('questions-words.txt') as file:
    lines = file.readlines()
    for line in lines:
        if re.match(r': currency',line):
            break

        line = line.replace('\n','').split(' ')
        if len(line)>2:
            countries.append(line[1])

countries = list(set(countries))
vectors = [model.get_vector(country) for country in countries]
linkage_result = linkage(vectors, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linkage_result, labels=countries)
plt.xlabel('country')
plt.ylabel('distance')
plt.savefig('No68.png')
