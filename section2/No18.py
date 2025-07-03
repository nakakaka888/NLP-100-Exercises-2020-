import sys

"""
実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

with open(sys.argv[1]) as file:
    lines = file.readlines()

lines.sort(key= lambda n: n.split()[2], reverse=True)
for line in lines:
    print(line.replace('\n', ''))


"""
出力結果(最初の10行):
>python No18.py popular-names.txt 
Linda   F       99689   1947
James   M       9951    1911
Mildred F       9921    1913
Mary    F       9889    1886
Mary    F       9888    1887
John    M       9829    1900
Elizabeth       F       9708    2012
Anna    F       9687    1913
Frances F       9677    1914
John    M       9655    1880

UNIXコマンド(最初の10行):
>sort -r -k 3 popular-names.txt 
Linda   F       99689   1947
James   M       9951    1911
Mildred F       9921    1913
Mary    F       9889    1886
Mary    F       9888    1887
John    M       9829    1900
Elizabeth       F       9708    2012
Anna    F       9687    1913
Frances F       9677    1914
John    M       9655    1880
"""
