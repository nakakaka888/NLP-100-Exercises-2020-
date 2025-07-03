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
        name_flag = False
        verb_flag = False
        str_src = ''
        str_dst = ''

        for morph in chunks[i].morphs:
            if morph.pos != '記号':
                if morph.pos == '名詞':
                    name_flag = True 
                str_src += morph.surface

        if int(chunks[i].dst) != -1:
            for morph in  chunks[int(chunks[i].dst)].morphs:
                if morph.pos != '記号':
                    if morph.pos == '動詞':
                        verb_flag = True
                    str_dst += morph.surface
                    
            if name_flag and verb_flag: 
                print('係り元:',str_src, '|係り先:',str_dst)
        
"""
>実行結果
係り元: 道具を |係り先: 用いて
係り元: 知能を |係り先: 研究する
係り元: 一分野を |係り先: 指す
係り元: 知的行動を |係り先: 代わって
係り元: 人間に |係り先: 代わって
係り元: コンピューターに |係り先: 行わせる
係り元: 研究分野とも |係り先: される
"""