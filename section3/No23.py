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

for line in article.splitlines():
    if re.search(r'^==*',line):
        level = len(re.match(r'^(=*)', line).group()) -1
        section = re.sub(r'[=\s]', '', line)
        print(section, level)

"""
出力(最初の10行):
>python No23.py
国名 1
歴史 1
地理 1
主要都市 2
気候 2
政治 1
元首 2
法 2
内政 2
地方行政区分 2
"""