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
import preprocessing

def train(epoch_num):
    path = "result.csv"
    df = pd.read_csv(path,dtype = 'object')

    mt = preprocessing.Mecab_neologd()

    m = model.LSTM(100,128,2)

    ohayou_words = ["おはよう","起床","起きた"]
    oyasumi_words = ["おやすみ","寝よう","寝る"]
    greetings = ohayou_words+oyasumi_words

    criterion = nn.CrossEntropyLoss()
    #２値分類だがrewardが離散なので平方２条誤差を用いる
    criterion2 = nn.MSELoss()

    word2vec = preprocessing.Word2vec()



    count = 0
    correct = 0

    x,y = df.shape
    LOG_FREQ = x

    for epoch in range(epoch_num):
        # for i in range(10):
        for i in range(df.shape[0]):
            begin_time = time.time()
            text = df.loc[i]["text"]
            ignore = False
            for greeting in greetings:
                if greeting in text:
                    ignore = True
                else:
                    text_normalized = mt.normalize_neologd(text)
                    wakati_texts = mt.m.parse(text_normalized).split(" ")
            if not ignore:
                inputs = np.array([])
                for word in wakati_texts:
                    try:
                        vec = word2vec.transform(word)
                        vec = np.reshape(vec,(1,1,-1))
                        if len(inputs)==0:
                            inputs = vec
                        else:
                            inputs = np.concatenate([inputs,vec])
                    except:
                        # print("Unexpected error:", sys.exc_info()[0])
                        pass
                        # raise

                inputs_ = torch.from_numpy(inputs)
                if inputs_.dim()<=2:
                    pass
                    # print("測定不可")
                else:
                    output = m(inputs_)
                    # print(output)
                    # print(output.shape)
                    predict = torch.argmax(output)
                    if(int(df.loc[i]["reply_favorited_count"])+int(df.loc[i]["reply_retweet_count"])>=1):
                        reward = torch.ones(1,dtype=torch.long)
                        reward2 = torch.from_numpy(np.array([0,1]))
                        label = 1

                    else:
                        reward = torch.zeros(1,dtype=torch.long)
                        reward2 = torch.from_numpy(np.array([1,0]))
                        label = 0

                    count += 1
                    if predict == label:
                        correct += 1

                    # print("output,label",output,reward)
                    # loss= criterion(output,reward)
                    loss = criterion2(output,reward2.float())
                    loss.backward()
                    m.optimizer.step()

                    # print("-"*15)
                    # print("epoch:{0},text:{1},time:{2:1f}".format(epoch+1,i+1,time.time()-begin_time))
                    # print("text:{0}".format(text))
                    # print("predict:{0},label:{1}".format(predict,label))
                    # print("loss:{0}".format(loss))

        print("正答率:{0}".format(correct/count))
        correct = 0
        count = 0


if __name__=="__main__":
    train(30)
