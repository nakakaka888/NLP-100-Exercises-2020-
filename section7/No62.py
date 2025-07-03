from gensim.models import KeyedVectors

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
p.47 縦の線を真っ直ぐ揃える
P.51 コードを段落に分割する
"""

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


most_similar =  model.most_similar(positive=['United_States'], topn=10)
for sim in most_similar:
    print(sim)

"""
<出力>
('Unites_States', 0.7877249121665955)
('Untied_States', 0.7541369795799255)
('United_Sates', 0.7400726079940796)
('U.S.', 0.7310774326324463)
('theUnited_States', 0.6404393315315247)
('America', 0.6178411841392517)
('UnitedStates', 0.6167312860488892)
('Europe', 0.6132988333702087)
('countries', 0.6044804453849792)
('Canada', 0.6019068956375122)
"""