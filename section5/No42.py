"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

class Chunk:
    def __init__(self,morphs,dst,src):
        self.morphs = morphs
        self.dst = dst 
        self.src = src

class Morph:
    def __init__(self, line):

        self.surface = line[0]
        info = line[1].split(',')
        self.base = info[6]
        self.pos = info[0]
        self.pos1 = info[1]

sentences = []
chunks = []
morphs = []

with open('ai.ja.txt.parsed') as f:
    lines = f.readlines()

for line in lines:
    line = line.split()
    if line[0] == '*': 
        if len(morphs)==0:
            dst = line[2].replace('D','')
            src = line[1]
        else:
            chunks.append(Chunk(morphs,dst,src))
            morphs = []
            dst = line[2].replace('D','')
            src = line[1]
        
    elif line[0] != 'EOS':
        morphs.append(Morph(line))
    else: 
        if len(morphs) != 0:
            chunks.append(Chunk(morphs,dst,src))
            sentences.append(chunks)
            morphs = []
            chunks = []

for chunks in sentences[:2]:
    for i in range(len(chunks)):
        str_src = ''
        str_dst = ''
        for morph in chunks[i].morphs:
            if morph.pos != '記号':
                str_src += morph.surface
        if int(chunks[i].dst) == -1:
            print('係り元:',str_src,'|係り先:無し')
        else:
            for morph in  chunks[int(chunks[i].dst)].morphs:
                if morph.pos != '記号':
                    str_dst += morph.surface
            print('係り元:',str_src, '|係り先:',str_dst)
        
"""
>実行結果
係り元: 人工知能 |係り先:無し
係り元: 人工知能 |係り先: 語
係り元: じんこうちのう |係り先: 語
係り元: AI |係り先: エーアイとは
係り元: エーアイとは |係り先: 語
係り元: 計算 |係り先: という
係り元: という |係り先: 道具を
係り元: 概念と |係り先: 道具を
係り元: コンピュータ |係り先: という
係り元: という |係り先: 道具を
係り元: 道具を |係り先: 用いて
係り元: 用いて |係り先: 研究する
係り元: 知能を |係り先: 研究する
係り元: 研究する |係り先: 計算機科学
係り元: 計算機科学 |係り先: の
係り元: の |係り先: 一分野を
係り元: 一分野を |係り先: 指す
係り元: 指す |係り先: 語
係り元: 語 |係り先: 研究分野とも
係り元: 言語の |係り先: 推論
係り元: 理解や |係り先: 推論
係り元: 推論 |係り先: 問題解決などの
係り元: 問題解決などの |係り先: 知的行動を
係り元: 知的行動を |係り先: 代わって
係り元: 人間に |係り先: 代わって
係り元: 代わって |係り先: 行わせる
係り元: コンピューターに |係り先: 行わせる
係り元: 行わせる |係り先: 技術または
係り元: 技術または |係り先: 研究分野とも
係り元: 計算機 |係り先: コンピュータによる
係り元: コンピュータによる |係り先: 情報処理システムの
係り元: 知的な |係り先: 情報処理システムの
係り元: 情報処理システムの |係り先: 実現に関する
係り元: 設計や |係り先: 実現に関する
係り元: 実現に関する |係り先: 研究分野とも
係り元: 研究分野とも |係り先: される
係り元: される |係り先:無し
"""