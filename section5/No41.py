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
    for chunk in chunks:
        str = ''
        for morph in chunk.morphs:
            str += morph.surface
        print('文節:',str,'|','係り先:',chunk.dst)

"""
>実行結果
文節: 人工知能 | 係り先: -1
文節: 人工知能 | 係り先: 17
文節: （じんこうちのう、、 | 係り先: 17
文節: AI | 係り先: 3
文節: 〈エーアイ〉）とは、 | 係り先: 17
文節: 「『計算 | 係り先: 5
文節: （）』という | 係り先: 9
文節: 概念と | 係り先: 9
文節: 『コンピュータ | 係り先: 8
文節: （）』という | 係り先: 9
文節: 道具を | 係り先: 10
文節: 用いて | 係り先: 12
文節: 『知能』を | 係り先: 12
文節: 研究する | 係り先: 13
文節: 計算機科学 | 係り先: 14
文節: （）の | 係り先: 15
文節: 一分野」を | 係り先: 16
文節: 指す | 係り先: 17
文節: 語。 | 係り先: 34
文節: 「言語の | 係り先: 20
文節: 理解や | 係り先: 20
文節: 推論、 | 係り先: 21
文節: 問題解決などの | 係り先: 22
文節: 知的行動を | 係り先: 24
文節: 人間に | 係り先: 24
文節: 代わって | 係り先: 26
文節: コンピューターに | 係り先: 26
文節: 行わせる | 係り先: 27
文節: 技術」、または、 | 係り先: 34
文節: 「計算機 | 係り先: 29
文節: （コンピュータ）による | 係り先: 31
文節: 知的な | 係り先: 31
文節: 情報処理システムの | 係り先: 33
文節: 設計や | 係り先: 33
文節: 実現に関する | 係り先: 34
文節: 研究分野」とも | 係り先: 35
文節: される。 | 係り先: -1
"""