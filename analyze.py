import numpy as np
import pandas as pd


df = pd.read_csv("result.csv","r",delimiter=",")
print(df.shape)


ohayou_words = ["おはよう","起床","起きた"]
oyasumi_words = ["おやすみ","寝よう","寝る"]

greetings = ohayou_words+oyasumi_words

x,y = df.shape
count = 0
for i in range(x):
    for greeting in greetings:
        if greeting in df.text[i]:
            # print(df.text[i])
            count += 1
            break


print("ratio{0}".format(count/x))
print(df.isnull().sum())
