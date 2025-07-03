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

for k,v in info_dict.items():
    print(k,':',v)



"""
出力(最初の10行):
>python No25.py
|略名 : イギリス
|日本語国名 : グレートブリテン及び北アイルランド連合王国
|公式国名 : {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />
|国旗画像 : Flag of the United Kingdom.svg
|国章画像 : [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
|国章リンク : （[[イギリスの国章|国章]]）
|標語 : {{lang|fr|[[Dieu et mon droit]]}}<br />（[[フランス語]]:[[Dieu et mon droit|神と我が権利]]）
|国歌 : [[女王陛下万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />''神よ女王を護り賜え''<br />{{center|[[ファイル:United States Navy Band - God Save the Queen.ogg]]}}
|地図画像 : Europe-UK.svg
|位置画像 : United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
"""