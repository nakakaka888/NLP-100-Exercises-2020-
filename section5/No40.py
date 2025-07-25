"""
リーダブルコードで実践したこと
P.19 名前に情報を追加する
P.51 コードを段落に分割する
"""

class Morph:
    def __init__(self, line):

        self.surface = line[0]
        info = line[1].split(',')
        self.base = info[6]
        self.pos = info[0]
        self.pos1 = info[1]

sentences = []
morphs =[]

with open('ai.ja.txt.parsed') as f:
    lines = f.readlines()

for line in lines:
    line = line.split()
    if line[0] == '*': 
        continue
    elif line[0] != 'EOS':
        morphs.append(Morph(line))
    else: 
        if len(morphs) != 0:
            sentences.append(morphs)
            morphs = []

for sentence in sentences[:2]:
    for morph in sentence:
        print(vars(morph))    

"""
>実行結果
{'surface': '人工', 'base': '人工', 'pos': '名詞', 'pos1': '一般'}
{'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}
{'surface': '人工', 'base': '人工', 'pos': '名詞', 'pos1': '一般'}
{'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': 'じん', 'base': 'じん', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'こうち', 'base': 'こうち', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'のう', 'base': 'のう', 'pos': '助詞', 'pos1': '終助詞'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': 'AI', 'base': '*', 'pos': '名詞', 'pos1': '一般'}
{'surface': '〈', 'base': '〈', 'pos': '記号', 'pos1': '括弧開'}
{'surface': 'エーアイ', 'base': '*', 'pos': '名詞', 'pos1': '固有名詞'}
{'surface': '〉', 'base': '〉', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': '「', 'base': '「', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '『', 'base': '『', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '計算', 'base': '計算', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': '』', 'base': '』', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '概念', 'base': '概念', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '並立助詞'}
{'surface': '『', 'base': '『', 'pos': '記号', 'pos1': '括弧開'}
{'surface': 'コンピュータ', 'base': 'コンピュータ', 'pos': '名詞', 'pos1': '一般'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': '』', 'base': '』', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '道具', 'base': '道具', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '用い', 'base': '用いる', 'pos': '動詞', 'pos1': '自立'}
{'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}
{'surface': '『', 'base': '『', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}
{'surface': '』', 'base': '』', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '研究', 'base': '研究', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': 'する', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}
{'surface': '計算', 'base': '計算', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': '機', 'base': '機', 'pos': '名詞', 'pos1': '接尾'}
{'surface': '科学', 'base': '科学', 'pos': '名詞', 'pos1': '一般'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}
{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'}
{'surface': '分野', 'base': '分野', 'pos': '名詞', 'pos1': '一般'}
{'surface': '」', 'base': '」', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '指す', 'base': '指す', 'pos': '動詞', 'pos1': '自立'}
{'surface': '語', 'base': '語', 'pos': '名詞', 'pos1': '一般'}
{'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}
{'surface': '「', 'base': '「', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '言語', 'base': '言語', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}
{'surface': '理解', 'base': '理解', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': 'や', 'base': 'や', 'pos': '助詞', 'pos1': '並立助詞'}
{'surface': '推論', 'base': '推論', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': '問題', 'base': '問題', 'pos': '名詞', 'pos1': 'ナイ形容詞語幹'}
{'surface': '解決', 'base': '解決', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': 'など', 'base': 'など', 'pos': '助詞', 'pos1': '副助詞'}
{'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}
{'surface': '知的', 'base': '知的', 'pos': '名詞', 'pos1': '一般'}
{'surface': '行動', 'base': '行動', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '人間', 'base': '人間', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'に', 'base': 'に', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '代わっ', 'base': '代わる', 'pos': '動詞', 'pos1': '自立'}
{'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}
{'surface': 'コンピューター', 'base': 'コンピューター', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'に', 'base': 'に', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '行わ', 'base': '行う', 'pos': '動詞', 'pos1': '自立'}
{'surface': 'せる', 'base': 'せる', 'pos': '動詞', 'pos1': '接尾'}
{'surface': '技術', 'base': '技術', 'pos': '名詞', 'pos1': '一般'}
{'surface': '」', 'base': '」', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': 'または', 'base': 'または', 'pos': '接続詞', 'pos1': '*'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': '「', 'base': '「', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '計算', 'base': '計算', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': '機', 'base': '機', 'pos': '名詞', 'pos1': '接尾'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': 'コンピュータ', 'base': 'コンピュータ', 'pos': '名詞', 'pos1': '一般'}
{'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'による', 'base': 'による', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '知的', 'base': '知的', 'pos': '名詞', 'pos1': '形容動詞語幹'}
{'surface': 'な', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}
{'surface': '情報処理', 'base': '情報処理', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'システム', 'base': 'システム', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}
{'surface': '設計', 'base': '設計', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': 'や', 'base': 'や', 'pos': '助詞', 'pos1': '並立助詞'}
{'surface': '実現', 'base': '実現', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': 'に関する', 'base': 'に関する', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '研究', 'base': '研究', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': '分野', 'base': '分野', 'pos': '名詞', 'pos1': '一般'}
{'surface': '」', 'base': '」', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': 'も', 'base': 'も', 'pos': '助詞', 'pos1': '係助詞'}
{'surface': 'さ', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}
{'surface': 'れる', 'base': 'れる', 'pos': '動詞', 'pos1': '接尾'}
{'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}
"""