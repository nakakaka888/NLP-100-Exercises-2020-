from gensim.models import KeyedVectors
import re
from sklearn.cluster import KMeans


"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

model = KeyedVectors.load_word2vec_format('./section7/GoogleNews-vectors-negative300.bin.gz', binary=True)
countries = []

with open('./section7/questions-words.txt') as file:
    lines = file.readlines()
    for line in lines:
        if re.match(r': currency',line):
            break

        line = line.replace('\n','').split(' ')
        if len(line)>2:
            countries.append(line[1])

countries = list(set(countries))
vectors = [model.get_vector(country) for country in countries]
kmeans_model = KMeans(n_clusters = 5).fit(vectors)

result = []
for country, label in zip(countries,kmeans_model.labels_):
    result.append([country,label])

result.sort(key=lambda x:x[1])
for info in result:
    print('国名:',info[0],' クラスタ:',info[1])


"""
>実行結果
国名: Nepal  クラスタ: 0
国名: Dominica  クラスタ: 0
国名: England  クラスタ: 0
国名: Bangladesh  クラスタ: 0
国名: Suriname  クラスタ: 0
国名: Tajikistan  クラスタ: 0
国名: Guyana  クラスタ: 0
国名: Jamaica  クラスタ: 0
国名: Bhutan  クラスタ: 0
国名: Pakistan  クラスタ: 0
国名: Georgia  クラスタ: 1
国名: Slovenia  クラスタ: 1
国名: Germany  クラスタ: 1
国名: Kyrgyzstan  クラスタ: 1
国名: Slovakia  クラスタ: 1
国名: Albania  クラスタ: 1
国名: Greece  クラスタ: 1
国名: Azerbaijan  クラスタ: 1
国名: Austria  クラスタ: 1
国名: Belarus  クラスタ: 1
国名: Italy  クラスタ: 1
国名: Poland  クラスタ: 1
国名: Kazakhstan  クラスタ: 1
国名: Sweden  クラスタ: 1
国名: Bulgaria  クラスタ: 1
国名: Switzerland  クラスタ: 1
国名: Montenegro  クラスタ: 1
国名: Liechtenstein  クラスタ: 1
国名: Malta  クラスタ: 1
国名: Turkey  クラスタ: 1
国名: Portugal  クラスタ: 1
国名: Belgium  クラスタ: 1
国名: Russia  クラスタ: 1
国名: Moldova  クラスタ: 1
国名: Hungary  クラスタ: 1
国名: Greenland  クラスタ: 1
国名: Armenia  クラスタ: 1
国名: Turkmenistan  クラスタ: 1
国名: Spain  クラスタ: 1
国名: Macedonia  クラスタ: 1
国名: Estonia  クラスタ: 1
国名: Croatia  クラスタ: 1
国名: Lithuania  クラスタ: 1
国名: Norway  クラスタ: 1
国名: Finland  クラスタ: 1
国名: Ireland  クラスタ: 1
国名: Cyprus  クラスタ: 1
国名: Uzbekistan  クラスタ: 1
国名: Serbia  クラスタ: 1
国名: Denmark  クラスタ: 1
国名: Romania  クラスタ: 1
国名: Latvia  クラスタ: 1
国名: France  クラスタ: 1
国名: Ukraine  クラスタ: 1
国名: Liberia  クラスタ: 2
国名: Nigeria  クラスタ: 2
国名: Zambia  クラスタ: 2
国名: Mali  クラスタ: 2
国名: Guinea  クラスタ: 2
国名: Zimbabwe  クラスタ: 2
国名: Namibia  クラスタ: 2
国名: Niger  クラスタ: 2
国名: Gabon  クラスタ: 2
国名: Mauritania  クラスタ: 2
国名: Rwanda  クラスタ: 2
国名: Angola  クラスタ: 2
国名: Mozambique  クラスタ: 2
国名: Botswana  クラスタ: 2
国名: Madagascar  クラスタ: 2
国名: Uganda  クラスタ: 2
国名: Gambia  クラスタ: 2
国名: Senegal  クラスタ: 2
国名: Malawi  クラスタ: 2
国名: Ghana  クラスタ: 2
国名: Kenya  クラスタ: 2
国名: Burundi  クラスタ: 2
国名: China  クラスタ: 3
国名: Peru  クラスタ: 3
国名: Taiwan  クラスタ: 3
国名: Qatar  クラスタ: 3
国名: Cuba  クラスタ: 3
国名: Japan  クラスタ: 3
国名: Oman  クラスタ: 3
国名: Ecuador  クラスタ: 3
国名: Australia  クラスタ: 3
国名: Bahamas  クラスタ: 3
国名: Canada  クラスタ: 3
国名: Honduras  クラスタ: 3
国名: Samoa  クラスタ: 3
国名: Uruguay  クラスタ: 3
国名: Tuvalu  クラスタ: 3
国名: Morocco  クラスタ: 3
国名: Indonesia  クラスタ: 3
国名: Bahrain  クラスタ: 3
国名: Jordan  クラスタ: 3
国名: Venezuela  クラスタ: 3
国名: Vietnam  クラスタ: 3
国名: Belize  クラスタ: 3
国名: Philippines  クラスタ: 3
国名: Fiji  クラスタ: 3
国名: Thailand  クラスタ: 3
国名: Laos  クラスタ: 3
国名: Chile  クラスタ: 3
国名: Nicaragua  クラスタ: 3
国名: Iran  クラスタ: 4
国名: Sudan  クラスタ: 4
国名: Libya  クラスタ: 4
国名: Algeria  クラスタ: 4
国名: Eritrea  クラスタ: 4
国名: Tunisia  クラスタ: 4
国名: Lebanon  クラスタ: 4
国名: Afghanistan  クラスタ: 4
国名: Iraq  クラスタ: 4
国名: Somalia  クラスタ: 4
国名: Egypt  クラスタ: 4
国名: Syria  クラスタ: 4
"""
    