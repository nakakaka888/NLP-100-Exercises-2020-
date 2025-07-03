import sys
import collections

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

for word, count in sorted(collections.Counter(list(head_list)).items(), key = lambda value: value[1], reverse=True):
    print(count, word)


"""
出力結果(最初の10行):
>python No19.py popular-names.txt 
118 James
111 William
108 John
108 Robert
92 Mary
75 Charles
74 Michael
73 Elizabeth
70 Joseph
60 Margaret


UNIXコマンド(最初の10行):
>cut -f 1 popular-names.txt  | sort | uniq -c | sort -r 
    118 James
    111 William
    108 Robert
    108 John
     92 Mary
     75 Charles
     74 Michael
     73 Elizabeth
     70 Joseph
     60 Margaret
"""
