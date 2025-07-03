import json
import gzip
import re


"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

dicts = []
with gzip.open('jawiki-country.json.gz', 'rb') as f:
    for line in f:
        dicts.append(json.loads(line))

for dict in dicts:
    if dict['title'] == 'イギリス':
        article = dict['text']

categories = []
for line in article.splitlines():
    if re.search(r'\[Category', line):
        categories.append(line)
        print(line)

"""
出力:
>python No31.py
[[Category:イギリス|*]]
[[Category:イギリス連邦加盟国]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]
"""