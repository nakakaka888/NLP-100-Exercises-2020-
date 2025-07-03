import re

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

file_name = 'neko.txt.mecab'

with open(file_name, 'r') as f:
    lines = f.read().split('EOS\n')


result_lists = []
for line in lines:
    lines_list = []
    for word in line.split('\n'):
        morpheme_list = re.split('\t|,',word)
        if len(morpheme_list)!=0 and morpheme_list[0]!='':
            morpheme_dict = {
                'surface':morpheme_list[0],
                'base'   :morpheme_list[7],
                'pos'    :morpheme_list[1],
                'pos1'   :morpheme_list[2]
            }
            lines_list.append(morpheme_dict)
    if lines_list:
        result_lists.append(lines_list)



verbase_list = [dict['base'] for dicts in result_lists for dict in dicts if dict['pos'] == '動詞']

print(verbase_list[:5])

"""
>実行結果
['生れる', 'つく', 'する', '泣く', 'する']
"""