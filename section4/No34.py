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


consname_list = []


for dicts in result_lists:
    temp_list = []
    for dict in dicts:
        if dict['pos']=='名詞':
            temp_list.append(dict['surface'])
        
        elif len(temp_list)>1:
            consname_list.append(temp_list)
            temp_list = []

        else:
            temp_list = []

    if len(temp_list)>1:
        consname_list.append(temp_list)

print(consname_list[:11])

"""
>実行結果
[['人間', '中'], ['一番', '獰悪'], ['時', '妙'], ['一', '毛'], ['その後', '猫'], ['一', '度'], ['ぷうぷうと', '煙'], ['邸', '内'], ['三', '毛'], ['書生', '以外'], ['四', '五', '遍']]
"""