import sys

"""
実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

with open(sys.argv[1]) as file:
    lines = file.readlines()

for line in lines[:int(sys.argv[2])]:
    print(line.rstrip())

"""
出力結果:
>python No14.py popular-names.txt 5
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880
Elizabeth       F       1939    1880
Minnie  F       1746    1880

UNIXコマンド:
>head -n 5 popular-names.txt
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880
Elizabeth       F       1939    1880
Minnie  F       1746    1880
"""
