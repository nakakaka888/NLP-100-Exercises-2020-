import sys

"""
実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

head_list = []
with open(sys.argv[1]) as file:
    lines = file.readlines()
    for line in lines:
        head_list.append(line.split()[0])

for word in sorted(list(set(head_list))):
    print(word)

"""
出力結果(最初の10行):
>python No17.py popular-names.txt 
Abigail
Aiden
Alexander
Alexis
Alice
Amanda
Amelia
Amy
Andrew
Angela

UNIXコマンド(最初の10行):
>cut -f 1 popular-names.txt | sort | uniq
Abigail
Aiden
Alexander
Alexis
Alice
Amanda
Amelia
Amy
Andrew
Angela
"""

