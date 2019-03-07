# encoding: utf8
from __future__ import unicode_literals
import copy
import numpy as np
import random
import pandas as pd
import MeCab
import gensim
import re
import unicodedata
import time

    # word2vecの教師モデルは乾けんきゅうしつのものを用いた[https://github.com/singletongue/WikiEntVec]
    # 導入方法について[https://www.tech-tech.xyz/machine-leaning-word2vec.html]
class Word2vec():
    def __init__(self):
        self.model =  gensim.models.KeyedVectors.load_word2vec_format("jawiki.all_vectors.100d.txt")

    def transform(self,word):
        return self.model[word]


# 参考にしたmecab dictionary[https://github.com/neologd/mecab-ipadic-neologd/blob/master/README.ja.md]
class Mecab_neologd():
    def __init__(self):
        self.m = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")

    def unicode_normalize(self,cls, s):
        pt = re.compile('([{}]+)'.format(cls))

        def norm(c):
            return unicodedata.normalize('NFKC', c) if pt.match(c) else c

        s = ''.join(norm(x) for x in re.split(pt, s))
        s = re.sub('－', '-', s)
        return s

    def remove_extra_spaces(self,s):
        s = re.sub('[ 　]+', ' ', s)
        blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                          '\u3040-\u309F',  # HIRAGANA
                          '\u30A0-\u30FF',  # KATAKANA
                          '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                          '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                          ))
        basic_latin = '\u0000-\u007F'

        def remove_space_between(cls1, cls2, s):
            p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
            while p.search(s):
                s = p.sub(r'\1\2', s)
            return s

        s = remove_space_between(blocks, blocks, s)
        s = remove_space_between(blocks, basic_latin, s)
        s = remove_space_between(basic_latin, blocks, s)
        return s

    def normalize_neologd(self,s):
        s = s.strip()
        s = self.unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

        def maketrans(f, t):
            return {ord(x): ord(y) for x, y in zip(f, t)}

        s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
        s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
        s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
        s = s.translate(
            maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                  '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

        s = self.remove_extra_spaces(s)
        s = self.unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
        s = re.sub('[’]', '\'', s)
        s = re.sub('[”]', '"', s)
        return s
    
def main():
    begin_time = time.time()
    path = "result.csv"
    df = pd.read_csv(path,dtype = 'object')
    text = df.loc[0]["text"]


    mt = Mecab_neologd()
    print('{0:01f}'.format(time.time()-begin_time))
    text_normalized = mt.normalize_neologd(text)
    print('{0:01f}'.format(time.time()-begin_time))
    wakati_text = mt.m.parse(text_normalized).split(" ")
    print('{0:01f}'.format(time.time()-begin_time))

    print(wakati_text)
    word2vec = Word2vec()
    for word in wakati_text:
        try:
            print(type(word2vec.transform(word)))
        except:
            print("error")




    # print(mt.parse(text))



if __name__=="__main__":
    main()
