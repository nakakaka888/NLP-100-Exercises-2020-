import sys

"""
実践したこと
P.19 名前に情報を追加する
"""

with open(sys.argv[1]) as file:
    file = file.read()
    file = file.replace('\t', ' ')
    print(file)

"""
出力結果(最初の10行):
> python No11.py popular-names.txt
Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
Elizabeth F 1939 1880
Minnie F 1746 1880
Margaret F 1578 1880
Ida F 1472 1880
Alice F 1414 1880
Bertha F 1320 1880
Sarah F 1288 1880

UNIXコマンド(最初の10行):
expand -t 1 popular-names.txt
>Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
Elizabeth F 1939 1880
Minnie F 1746 1880
Margaret F 1578 1880
Ida F 1472 1880
Alice F 1414 1880
Bertha F 1320 1880
Sarah F 1288 1880
"""

#UNIXコマンド: expand -t タブの幅指定　popular-names.txt
