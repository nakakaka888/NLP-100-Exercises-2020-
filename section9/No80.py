import pandas as pd

train_x = pd.read_csv('./section9/train.txt', sep='\t') 

def data_modify(data):
    data_x = data['TITLE']
    data_x = data_x.str.replace(r'\'s|[\'\"\:\.,\;\!\&\?\$]', '', regex=True)
    data_x = data_x.str.replace(r'\s-\s', ' ', regex=True)
    data_x = data_x.str.lower()

    sentence = [line.split() for line in data_x]
    
    return sentence

def word_count(sentence):
    word_list = {}
    for list in sentence:
        for word in list:
            if word not in word_list:
                word_list[word] = 1
            else:
                word_list[word] += 1

    word_list = sorted(word_list.items(), reverse=True, key = lambda x : x[1])

    rank_list = {}
    rank = 1
    for i, (item,key) in enumerate(word_list):

        if key < 2:
            rank_list[item] = 0
        else:
            rank_list[item] = rank
            rank += 1
    return rank_list
    


def get_id(words, rank_list):
    for word in words:
        id = rank_list[word]
        print(f"word:{word}, id:{id}")


sentence = data_modify(train_x)
rank_list = word_count(sentence)


for words in sentence[:2]:
    get_id(words, rank_list)
    print("\n")


"""
>実行結果
word:janet, id:3184
word:yellen, id:148
word:stakes, id:4468
word:out, id:41
word:position, id:3705
word:on, id:6
word:fighting, id:2002
word:financial, id:814
word:stability, id:3706
word:risks, id:952


word:update, id:8
word:1-outkast, id:5684
word:goes, id:363
word:back, id:127
word:to, id:1
word:1990s, id:4469
word:hip, id:2217
word:hop, id:2465
word:at, id:13
word:coachella, id:778
word:reunion, id:1394


"""