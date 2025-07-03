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
            cases = []
            for i in range(len(chunks)):
                if chunks[i].dst == verb[1]:
                    for morph in chunks[i].morphs:
                        if morph.pos == '助詞':
                            cases.append(morph.surface)
            cases.sort()
            corpus.append(verb[0]+' '+' '.join(cases))

for i, line in enumerate(corpus):
    print(line, i)

with open('./section5/No45.txt', mode='w') as fw:
    for line in corpus:
        fw.write(line + '\n')



"""
>実行結果
No45.txt

UNIXコマンド
>sort ./section5/No45.txt |uniq -c|sort -r
    49 する を
    18 する が
    15 する に
    14 する と
    12 する は を

>grep "行う" ./section5/No45.txt |sort |uniq -c| sort -r
    8 行う を
    1 行う まで を
    1 行う は を をめぐって
    1 行う は を
    1 行う に を を
    
>grep "なる" ./section5/No45.txt |sort |uniq -c| sort -r
    3 なる に は
    3 なる が と
    2 なる に
    2 なる と
    1 異なる も
    
>grep "与える" ./section5/No45.txt |sort |uniq -c| sort -r
    1 与える に は を
    1 与える が に
    1 与える が など に
"""