"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

from graphviz import Digraph

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

dot = Digraph('round-table', format='png', comment='The Round Table') 
family_list = []

chunks = sentences[1]

for i in range(len(chunks)):
    str_src = ''
    str_dst = ''
    for morph in chunks[i].morphs:
        if morph.pos != '記号':
            str_src += morph.surface
    dot.node(chunks[i].src,str_src) 

    if int(chunks[i].dst) == -1:
        continue
    else:
        family_list.append([chunks[i].src, chunks[i].dst])

for pair in family_list:
    dot.edge(pair[0], pair[1])

dot.render('./section5/example1', view=True)

"""
>実行結果
example1.png
"""
