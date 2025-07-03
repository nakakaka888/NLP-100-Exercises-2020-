from gensim.models import KeyedVectors

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
vectors = []

with open('questions-words.txt') as file:
    lines = file.readlines()
    for line in lines:
        vectors.append(line.replace('\n','').split(' '))

vectors = vectors[:20]


for i in range(len(vectors)):
    if len(vectors[i])>2:
        most_similar = model.most_similar(positive=[vectors[i][1],vectors[i][2]], negative=[vectors[i][0]], topn=1) 
        vectors[i].extend(list(most_similar[0]))
        print(vectors[i])

"""
>実行結果
['Athens', 'Greece', 'Baghdad', 'Iraq', 'Iraqi', 0.635187029838562]
['Athens', 'Greece', 'Bangkok', 'Thailand', 'Thailand', 0.7137669920921326]
['Athens', 'Greece', 'Beijing', 'China', 'China', 0.7235777378082275]
['Athens', 'Greece', 'Berlin', 'Germany', 'Germany', 0.673462450504303]
['Athens', 'Greece', 'Bern', 'Switzerland', 'Switzerland', 0.4919748306274414]
['Athens', 'Greece', 'Cairo', 'Egypt', 'Egypt', 0.7527808547019958]
['Athens', 'Greece', 'Canberra', 'Australia', 'Australia', 0.5837326049804688]
['Athens', 'Greece', 'Hanoi', 'Vietnam', 'Viet_Nam', 0.6276342272758484]
['Athens', 'Greece', 'Havana', 'Cuba', 'Cuba', 0.6460990905761719]
['Athens', 'Greece', 'Helsinki', 'Finland', 'Finland', 0.6899983286857605]
['Athens', 'Greece', 'Islamabad', 'Pakistan', 'Pakistan', 0.7233325839042664]
['Athens', 'Greece', 'Kabul', 'Afghanistan', 'Afghan', 0.6160915493965149]
['Athens', 'Greece', 'London', 'England', 'Britain', 0.5646187663078308]
['Athens', 'Greece', 'Madrid', 'Spain', 'Spain', 0.7036614418029785]
['Athens', 'Greece', 'Moscow', 'Russia', 'Russia', 0.7382972240447998]
['Athens', 'Greece', 'Oslo', 'Norway', 'Norway', 0.6470743417739868]
['Athens', 'Greece', 'Ottawa', 'Canada', 'Canada', 0.5912168622016907]
['Athens', 'Greece', 'Paris', 'France', 'France', 0.672462522983551]
['Athens', 'Greece', 'Rome', 'Italy', 'Italy', 0.6826189756393433]
"""

