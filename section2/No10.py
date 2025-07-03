import sys

"""
実践したこと
P.19 名前に情報を追加する

"""

file = open(sys.argv[1])
lines = file.readlines()
print(len(lines))
file.close()

"""
出力結果:
> python No10.py popular-names.txt
2780

UNIXコマンド:
wc -l popular-names.txt
>2780 popular-names.txt
"""
