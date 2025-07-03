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

count_word = {}

for dicts in result_lists:
    for dict in dicts:
        if dict['surface'] not in count_word:
            count_word[dict['surface']] = 1
        else:
            count_word[dict['surface']] += 1

count_word = sorted(count_word.items(), reverse=True, key = lambda x : x[1])

for i in range(11):
    print(count_word[i])

"""
>実行結果
('の', 9194)
('。', 7486)
('て', 6868)
('、', 6772)
('は', 6420)
('に', 6243)
('を', 6071)
('と', 5508)
('が', 5337)
('た', 3988)
('で', 3806)
"""
