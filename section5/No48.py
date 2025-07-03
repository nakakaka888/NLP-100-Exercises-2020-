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

corpus = []
name_flag = False

for chunks in sentences[:2]:
    verb_list = []
    for i in range(len(chunks)):
        phrase = ''
        for morph in chunks[i].morphs:
            if morph.pos != '記号':
                phrase += morph.surface
            if morph.pos =='名詞':
                name_flag = True
        
        if name_flag:
            name_flag = False
            dst = chunks[i].dst
            while dst != '-1':
                phrase += '-> '
                for morph in chunks[int(dst)].morphs:
                    if morph.pos != '記号':
                        phrase += morph.surface
                dst = chunks[int(dst)].dst
            print(phrase)

"""
>実行結果
人工知能
人工知能-> 語-> 研究分野とも-> される
じんこうちのう-> 語-> 研究分野とも-> される
AI-> エーアイとは-> 語-> 研究分野とも-> される
エーアイとは-> 語-> 研究分野とも-> される
計算-> という-> 道具を-> 用いて-> 研究する-> 計算機科学-> の-> 一分野を-> 指す-> 語-> 研究分野とも-> される
概念と-> 道具を-> 用いて-> 研究する-> 計算機科学-> の-> 一分野を-> 指す-> 語-> 研究分野とも-> される
コンピュータ-> という-> 道具を-> 用いて-> 研究する-> 計算機科学-> の-> 一分野を-> 指す-> 語-> 研究分野とも-> される
道具を-> 用いて-> 研究する-> 計算機科学-> の-> 一分野を-> 指す-> 語-> 研究分野とも-> される
知能を-> 研究する-> 計算機科学-> の-> 一分野を-> 指す-> 語-> 研究分野とも-> される
研究する-> 計算機科学-> の-> 一分野を-> 指す-> 語-> 研究分野とも-> される
計算機科学-> の-> 一分野を-> 指す-> 語-> 研究分野とも-> される
一分野を-> 指す-> 語-> 研究分野とも-> される
語-> 研究分野とも-> される
言語の-> 推論-> 問題解決などの-> 知的行動を-> 代わって-> 行わせる-> 技術または-> 研究分野とも-> される
理解や-> 推論-> 問題解決などの-> 知的行動を-> 代わって-> 行わせる-> 技術または-> 研究分野とも-> される
推論-> 問題解決などの-> 知的行動を-> 代わって-> 行わせる-> 技術または-> 研究分野とも-> される
問題解決などの-> 知的行動を-> 代わって-> 行わせる-> 技術または-> 研究分野とも-> される
知的行動を-> 代わって-> 行わせる-> 技術または-> 研究分野とも-> される
人間に-> 代わって-> 行わせる-> 技術または-> 研究分野とも-> される
コンピューターに-> 行わせる-> 技術または-> 研究分野とも-> される
技術または-> 研究分野とも-> される
計算機-> コンピュータによる-> 情報処理システムの-> 実現に関する-> 研究分野とも-> される
コンピュータによる-> 情報処理システムの-> 実現に関する-> 研究分野とも-> される
知的な-> 情報処理システムの-> 実現に関する-> 研究分野とも-> される
情報処理システムの-> 実現に関する-> 研究分野とも-> される
設計や-> 実現に関する-> 研究分野とも-> される
実現に関する-> 研究分野とも-> される
研究分野とも-> される
"""