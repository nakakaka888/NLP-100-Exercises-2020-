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

for line in categories:
    print(re.findall(r'\[\[Category:(.*)\]\]', line).pop())

"""
出力:
>python No31.py
イギリス|*
イギリス連邦加盟国
英連邦王国|*
G8加盟国
欧州連合加盟国|元
海洋国家
現存する君主国
島国
1801年に成立した国家・領域
"""