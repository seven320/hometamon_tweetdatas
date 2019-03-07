#encoding utf-8

import datetime
import tweepy
import random

import pandas as pd
import MeCab
import csv
import copy
#親ディレクトリにあるアカウント情報へのパス
import sys,os
pardir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

#account情報をaccount.pyからロード
from account import account #load account
# from metamon_code import meta_manuscript

class Hometamon():
    def __init__(self,test = True):
        auth = account.Initialize()
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        self.twitter_id = account.id()
        # self.manuscript = meta_manuscript.Manuscript()
        JST = datetime.timezone(datetime.timedelta(hours=+9),"JST")
        self.jst_now = datetime.datetime.now(JST)
        #for test
        self.test = test

    def check_timeline(self):
        since = None
        public_tweets = self.api.home_timeline(count=10,since_id=since)
        print(self.twitter_id)
        for tweet in public_tweets:
            print("-"*20)
            # print(tweet)
            print(tweet.text)
            #RTには返事しない
            tweet_split = tweet.text.split(" ")
            if tweet_split[0] == "RT":
                exclude = True
                print("this is retweeted")
            else:
                print("this is original")

            if tweet.user.screen_name == self.twitter_id:
                print("-"*10)
                print(tweet.user.name,tweet.user.screen_name,"\n",tweet.text)
            # print(tweet.user.name,tweet.user.screen_name)

    def gather_tweet_by_tweet_id(self,tweet_id = ["1061896865484959745"]):
        tweets = self.api.statuses_lookup(id_=tweet_id)

        # for tweet in tweets:
        #     print(tweet.user.name)
        #     print(tweet.user.screen_name)#@以下のID
        #     print(tweet.text)

        return tweets





def main():
# CSVファイルから必要なIDなどデータを引っ張ってくる
    path = "tweets.csv"
    tweets_csv = pd.read_csv(path,dtype = 'object')
    gather_tweet_ids = []
    gather_tweet_ids_lists = []
    hometamon_tweet_ids = []
    hometamon_tweet_ids_lists= []
    for tweet in tweets_csv.values:
        if tweet[1] != "nan":#返事ができてないやつを除外
            hometamon_tweet_ids.append(tweet[0])
            gather_tweet_ids.append(tweet[1])
        # test
        # if len(gather_tweet_ids) == 100:
        #     break
        if len(gather_tweet_ids) == 100:
            hometamon_tweet_ids_lists.append(hometamon_tweet_ids)
            gather_tweet_ids_lists.append(gather_tweet_ids)
            hometamon_tweet_ids = []
            gather_tweet_ids = []
    hometamon_tweet_ids_lists.append(hometamon_tweet_ids)
    gather_tweet_ids_lists.append(gather_tweet_ids)

#データを収集,書き込みする
    hometamon = Hometamon()

    result_columns = ["created_at","id","id_str","text","truncated","in_reply_to_status_id","in_reply_to_status_id_str","in_reply_to_user_id","in_reply_to_user_id_str","in_reply_to_screen_name","user","retweet_count","favorite_count","favorited","retweeted","lang",]


    with open("result.csv","w") as csvfile:
        writer = csv.writer(csvfile,delimiter=",")
        columns = copy.deepcopy(result_columns)
        columns.append("reply_retweet_count")
        columns.append("reply_favorited_count")
        writer.writerow(columns)

        for i in range(len(gather_tweet_ids_lists)):
            gather_tweets = hometamon.gather_tweet_by_tweet_id(gather_tweet_ids_lists[i])
            hometamon_tweets = hometamon.gather_tweet_by_tweet_id(hometamon_tweet_ids_lists[i])
            result_data = {}
            for j in range(len(gather_tweets)):
                for result_column in result_columns:
                    result_data[result_column] = gather_tweets[j]._json[result_column]
                result_data["reply_retweet_count"] = hometamon_tweets[j]._json["retweet_count"]
                result_data["reply_favorited_count"] = hometamon_tweets[j]._json["favorite_count"]
                writer.writerow(result_data.values())




if __name__ == "__main__":
    # hometamon = Hometamon()
    # tweets = hometamon.gather_tweet_by_tweet_id()
    main()
