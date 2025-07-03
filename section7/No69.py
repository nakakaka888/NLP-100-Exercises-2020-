from gensim.models import KeyedVectors
import re
from matplotlib import pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
import numpy as np 
from sklearn.cluster import KMeans

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


countries = set(countries)

vectors = [model.get_vector(country) for country in countries]
vectors = np.array(vectors)

kmeans_model = KMeans(n_clusters = 5).fit(vectors)

tsne = TSNE(n_components=2, random_state=42)
result_tsne = tsne.fit_transform(vectors)

color_list = ['r', 'g', 'b', 'y', 'orange']

plt.figure(figsize=(20, 15))
for vec, country, i in zip(result_tsne, countries, kmeans_model.labels_):
    plt.scatter(vec[0],vec[1], color = color_list[i])
    plt.annotate(country, xy=(vec[0], vec[1]))

plt.savefig('No69.png')
