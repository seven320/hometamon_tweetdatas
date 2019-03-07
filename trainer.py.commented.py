# encoding:utf-8
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import time
import sys,os

import model
import analyze

def train(epoch_num):
    path = "result.csv"  #_ str
    df = pd.read_csv(path,dtype = 'object')  #_ (2064, 18)

    mt = analyze.Mecab_neologd()  #_ Mecab_neologd

    m = model.LSTM(100,128,2)  #_ LSTM


    criterion = nn.CrossEntropyLoss()  #_ CrossEntropyLoss
    #２値分類だがrewardが離散なので平方２条誤差を用いる
    criterion2 = nn.MSELoss()  #_ MSELoss

    word2vec = analyze.Word2vec()  #_ Word2vec

    ohayou_words = ["おはよう","起床","起きた"]
    oyasumi_words = ["おやすみ","寝よう","寝る"]

    for epoch in range(epoch_num):
        for i in range(100):
        # for i in range(df.shape[0]):
            begin_time = time.time()  #_ float
            text = df.loc[i]["text"]  #_ str
            text_normalized = mt.normalize_neologd(text)  #_ str

            wakati_text = mt.m.parse(text_normalized).split(" ")  #_ [str,str,str,str,str,str,]

            inputs = np.array([])  #_ (0,)

            for word in wakati_text:
                try:
                    vec = word2vec.transform(word)  #_ (100,)
                    vec = np.reshape(vec,(1,1,-1))  #_ (1, 1, 100)
                    if len(inputs)==0:
                        inputs = vec  #_ (1, 1, 100)
                    else:
                        inputs = np.concatenate([inputs,vec])  #_ (5, 1, 100)
                except:
                    # print("Unexpected error:", sys.exc_info()[0])
                    pass
                    # raise
            #
            # print(inputs.shape)
            inputs_ = torch.from_numpy(inputs)  #_ torch.Size([5, 1, 100])
            if inputs_.dim()<3:
                print("測定不可")
            else:
                output = m(inputs_)  #_ torch.Size([2])
                # print(output)
                # print(output.shape)

                predict = torch.argmax(output)  #_ torch.Size([])






                if(int(df.loc[i]["reply_favorited_count"])+int(df.loc[i]["reply_retweet_count"])>=1):
                    reward = torch.ones(1,dtype=torch.long)  #_ torch.Size([1])
                    reward2 = torch.from_numpy(np.array([0,1]))  #_ torch.Size([2])

                else:
                    reward = torch.zeros(1,dtype=torch.long)  #_ torch.Size([1])
                    reward2 = torch.from_numpy(np.array([1,0]))  #_ torch.Size([2])

                print("output:{0},reward{1}".format(output,reward2.float()))

                loss = criterion2(output,reward2.float())  #_ torch.Size([])

                # try:loss=criterion2(predict,reward2)
                # except:pass

                loss.backward()
                m.optimizer.step()
                print("-"*15)
                print("epoch:{0},text:{1},time:{2:1f}".format(epoch+1,i+1,time.time()-begin_time))
                print("text:{0}".format(text))
                print("predict:{0}".format(predict))
                print(loss)

if __name__=="__main__":
    train(1)
