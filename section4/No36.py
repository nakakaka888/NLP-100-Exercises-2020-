import re
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

font_path = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams["font.family"] = font_prop.get_name()
matplotlib.use('Agg') # -----(1)

file_name = './section4/neko.txt.mecab'

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

#グラフの描画
keys = []
values = []
for i in range(10):
    keys.append(count_word[i][0])
    values.append(count_word[i][1])

plt.xlabel("頻出単語")
plt.ylabel("出現回数")
plt.bar(keys, values)
plt.savefig('./section4/1.png')