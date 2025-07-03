import re

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

file_name = 'neko.txt.mecab'

with open(file_name, 'r') as f:
    lines = f.read().split('EOS\n')


result_list = []
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
        result_list.append(lines_list)

for i in range(5):
    print(result_list[i])


"""
>実行結果
[{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'}]
[{'surface': '\u3000', 'base': '\u3000', 'pos': '記号', 'pos1': '空白'}, {'surface': '吾輩', 'base': '吾輩', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '猫', 'base': '猫', 'pos': '名詞', 'pos1': ' 一般'}, {'surface': 'で', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'ある', 'base': 'ある', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}]
[{'surface': '名前', 'base': '名前', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': 'まだ', 'base': 'まだ', 'pos': '副詞', 'pos1': '助詞類接続'}, {'surface': '無い', 'base': '無い', 'pos': '形容詞', 'pos1': '自立'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}]
[{'surface': '\u3000', 'base': '\u3000', 'pos': '記号', 'pos1': '空白'}, {'surface': 'どこ', 'base': 'どこ', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'で', 'base': 'で', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '生れ', 'base': '生れる', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'か', 'base': 'か', 'pos': '助詞', 'pos1': ' 副助詞／並立助詞／終助詞'}, {'surface': 'とんと', 'base': 'とんと', 'pos': '副詞', 'pos1': '一般'}, {'surface': '見当', 'base': '見当', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'が', 'base': 'が', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'つか', 'base': 'つく', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'ぬ', 'base': 'ぬ', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}]
[{'surface': '何', 'base': '何', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'でも', 'base': 'でも', 'pos': '助詞', 'pos1': '副助詞'}, {'surface': '薄暗い', 'base': '薄暗い', 'pos': '形容詞', 'pos1': '自立'}, {'surface': 'じめじめ', 'base': 'じめじめ', 'pos': '副 詞', 'pos1': '一般'}, {'surface': 'し', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動 詞', 'pos1': '*'}, {'surface': '所', 'base': '所', 'pos': '名詞', 'pos1': '非自立'}, {'surface': 'で', 'base': 'で', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'ニャーニャー', 'base': '*', 'pos': '名詞', 'pos1': '一般'}, {'surface': '泣い', 'base': '泣く', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': 'いた事', 'base': 'いた事', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'だけ', 'base': 'だけ', 'pos': '助詞', 'pos1': '副助詞'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '記憶', 'base': '記憶', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'し', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': 'いる', 'base': 'いる', 'pos': '動詞', 'pos1': '非自立'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}]
"""