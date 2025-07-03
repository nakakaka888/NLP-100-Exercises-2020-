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
case_flag = False
verb_falg = False


for chunks in sentences:
    verb_list = []
    for chunk in chunks:
        morphs = chunk.morphs
        name = ''
        for i in range(len(morphs)):
            if morphs[i].pos == '名詞' and morphs[i].pos1 == 'サ変接続' and i+2 <= (len(morphs)):
                if morphs[i+1].surface == 'を':
                    name = morphs[i].surface + morphs[i+1].surface
        if name:
            for sub_morph in chunks[int(chunk.dst)].morphs:
                if sub_morph.pos == '動詞':
                    verb_list.append([name + sub_morph.base, chunks[int(chunk.dst)].src])
                    chunk.dst = -1
                    break
        
    if verb_list:
        for verb in verb_list:
            family_dict = {}
            for i in range(len(chunks)):
                if chunks[i].dst == verb[1]:
                    vocab = ''
                    case = ''
                    for morph in chunks[i].morphs:
                        if morph.pos1 != '句点' and morph.pos1 != '読点':   
                            vocab += morph.surface
                        if morph.pos == '助詞':
                            case = morph.surface
                    if case:
                        family_dict[vocab] = case
            sorted_dict = dict(sorted(family_dict.items(), key=lambda item: item[1]))
            corpus.append(verb[0]+' '+' '.join(sorted_dict.values())+' '+' '.join(sorted_dict.keys()))

for line in corpus[:10]:
    print(line)

"""
>実行結果
行動を代わる に 人間に
判断をする  
処理を用いる  
記述をする と 主体と
注目を集める が 「サポートベクターマシン」が
経験を行う に 元に
学習を行う に 元に
流行を超える  
学習を繰り返す  
学習をする て に は を を通して なされている 元に ACT-Rでは 推論ルールを 生成規則を通して
"""