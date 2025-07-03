import sys

"""
実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

with open('col1.txt', mode='r') as file1, open('col2.txt', mode='r') as file2:
    line1 = file1.readlines()
    line2 = file2.readlines()

with open('merge.txt', mode='w') as fw:
    for merge1, merge2 in zip(line1,line2):
        fw.write(merge1.replace('\n', '\t') + merge2)

""" 
出力結果:
>python No13.py 
merge.txt参照

UNIXコマンド(最初の10行):
>paste col1.txt col2.txt
Mary    F
Anna    F
Emma    F
Elizabeth       F
Minnie  F
Margaret        F
Ida     F
Alice   F
Bertha  F
Sarah   F
"""


