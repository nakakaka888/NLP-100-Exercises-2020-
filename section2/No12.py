import sys

"""
実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

def write_extract_col(lines, number, path):
    col = ''
    for sentence in lines:
        word_list = sentence.split()
        col += word_list[number] + '\n'
    
    with open(path, mode='w') as file:
        file.write(col)

with open(sys.argv[1]) as file:
    lines = file.readlines()

write_extract_col(lines, 0, 'col1.txt')
write_extract_col(lines, 1, 'col2.txt')

"""
出力結果:
>python No12.py popular-names.txt
col1.txt,col2.txtファイル参照

UNIXコマンド(最初の10行):
>cut -f 1 popular-names.txt
Mary
Anna
Emma
Elizabeth
Minnie
Margaret
Ida
Alice
Bertha
Sarah
"""

