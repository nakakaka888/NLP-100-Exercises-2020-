import re

file_name = 'neko.txt.mecab'


"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""


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

AofB_lists = []

for dicts in result_lists:
    for i in range(1, len(dicts)-1):
        if dicts[i-1]['pos']=='名詞' and dicts[i]['base']=='の' and dicts[i+1]['pos']=='名詞':
            AofB_lists.append(dicts[i-1]['surface'] + ':'+ dicts[i+1]['surface'])
            

print(AofB_lists[:10])

"""
>実行結果
['彼:掌', '掌:上', '書生:顔', 'はず:顔', '顔:真中', '穴:中', '書生:掌', '掌:裏', '何:事', '肝心:母親']
"""