import sys

"""
実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

with open(sys.argv[1]) as file:
    lines = file.readlines()

split_n = int(sys.argv[2])

part_idx = (len(lines) + split_n -1) // split_n
result = ''
for i in range(part_idx):
    fn = lines[split_n * i: split_n * (i+1)]
    with open('split_f%s.txt' %i, mode='w') as fw:
        fw.write(''.join(fn))

"""
出力結果:
>python No16.py popular-names.txt 1000
split_f0.txt ~ split_f2.txtを参照

UNIXコマンド:
split -l 1000 popular-names.txt
"""

