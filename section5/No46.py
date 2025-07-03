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
for chunks in sentences:
    verb_list = []
    for i in range(len(chunks)):
        for morph in chunks[i].morphs:
             if morph.pos == '動詞':
                verb_list.append([morph.base,chunks[i].src])
                break
    if verb_list:
        for verb in verb_list:
            family_dict = {}
            for i in range(len(chunks)):
                if chunks[i].dst == verb[1]:
                    vocab = ''
                    case = ''
                    for morph in chunks[i].morphs:    
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
用いる を 道具を
する て を 用いて 『知能』を
指す を 一分野」を
代わる に を 人間に 知的行動を
行う て に 代わって コンピューターに
する も 研究分野」とも
述べる で に は 解説で、 次のように 佐藤理史は
する で を コンピュータ上で 知的能力を
する を 推論・判断を
する を 画像データを
"""
