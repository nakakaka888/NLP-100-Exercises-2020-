import json
import gzip
import re
import requests

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

lines = article.splitlines()
flag = False
info_list = []
info_dict = {}

for line in lines:
    if re.search(r'{{基礎情報',line):
        flag = True

    if flag:
        info_list.append(line)

    if re.match(r'}}', line):
        flag = False

for info in info_list:
    if re.match(r'\|', info):
        info_dict.setdefault(info.split('=',1)[0].strip(),info.split('=',1)[1].strip())

info_dict = {
               key : re.sub(r'\'\'+', '', value) for key, value in info_dict.items()
            }


def del_inlink(value):
    value = re.sub(r'\[\[[^\|\]]+\|[^\|\]]+\|([^\|\]]+)\]\]',r'\1', value) 
    value = re.sub(r'\[\[[^\|\]]+\|([^\]]+)\]\]', r'\1', value)
    value = re.sub(r'\[\[([^\]]+)\]\]',r'\1', value)
    return value

info_dict = {
               key : del_inlink(value) for key, value in info_dict.items()
            }

def del_markup(value):
    value = re.sub(r'\{\{[^\|]+\|[^\|]+\|([^\}]+)\}\}', r'\1', value)
    value = re.sub(r'\{\{[^\|]+\|([^\}]+)\}\}', r'\1', value)
    value = re.sub(r'\[.*\]', '', value)
    value = re.sub(r'\{\{0\}\}', '', value)
    return value

info_dict = {
               key : del_markup(value) for key, value in info_dict.items()
            }

URL = "https://kousokuwiki.org/w/api.php"

S = requests.Session()

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "File:"+info_dict['|国旗画像'],
    "iiprop": "url"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

PAGES = DATA["query"]["pages"]


for k, v in PAGES.items():
    print(v["imageinfo"][0]["url"])


"""
出力(最初の10行):
>python No29.py
https://upload.wikimedia.org/wikipedia/commons/8/83/Flag_of_the_United_Kingdom_%283-5%29.svg
"""