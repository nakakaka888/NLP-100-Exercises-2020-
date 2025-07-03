import sys

"""
実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

with open(sys.argv[1]) as file:
    lines = file.readlines()

for line in lines[int(sys.argv[2])*-1:]:
   print(line.rstrip())

"""
出力結果:
>python No15.py popular-names.txt 5
Benjamin        M       13381   2018
Elijah  M       12886   2018
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018

UNIXコマンド:
>tail -n 5 popular-names.txt
Benjamin        M       13381   2018
Elijah  M       12886   2018
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018
"""
