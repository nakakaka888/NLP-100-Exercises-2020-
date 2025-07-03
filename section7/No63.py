from gensim.models import KeyedVectors

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
most_similar =  model.most_similar(positive=['Spain','Madrid'], negative=['Athens'], topn=10)

for sim in most_similar:
    print(sim)

"""
<出力>
('Spains', 0.6056791543960571)
('Barcelona', 0.6044400334358215)
('Spaniards', 0.5837481617927551)
('Málaga', 0.5805598497390747)
('Malaga', 0.5797936916351318)
('Spanish', 0.5793157815933228)
('Catalan', 0.5683084726333618)
('San_Sebastián', 0.5657953023910522)
('Salave_Gold_Deposit', 0.5624397993087769)
('Inveravante_Inversiones_SL', 0.5606337785720825)
"""