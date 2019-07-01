## Obtained from Sem/eval 2018 website
import os,sys,inspect
import time
import datetime
import argparse

from twitter import *

parser = argparse.ArgumentParser(description="downloads tweets")
parser.add_argument('--partial', dest='partial', default=None, type=argparse.FileType('r'))
parser.add_argument('--dist', dest='dist', default=None, type=argparse.FileType('r'), required=True)
parser.add_argument('--output', dest='output', default=None, type=argparse.FileType('w'), required=True)
args = parser.parse_args()


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import twitterkeys

consumer_key = twitterkeys.CONSUMER_KEY
consumer_secret   = twitterkeys.CONSUMER_SECRET
access_token   = twitterkeys.ACCESS_TOKEN
access_token_secret   = twitterkeys.ACCESS_TOKEN_SECRET

t = Twitter(auth=OAuth(access_token, access_token_secret, consumer_key, consumer_secret))

cache = {}
if args.partial != None:
    for line in args.partial:
        fields = line.strip().split("\t")
        text = fields[-1]
        sid = fields[0]
        cache[sid] = text.encode('utf-8')

for line in args.dist:
    fields = line.strip().split('\t')
    sid = fields[0]

    while not sid in cache:
        try:
            temp_var = t.statuses.show(_id=sid, tweet_mode='extended')
            if 'retweeted_status' in temp_var.keys():
                text = temp_var['retweeted_status']['full_text'].replace('\n', ' ').replace('\r', ' ')
            else:
                text = temp_var['full_text'].replace('\n', ' ').replace('\r', ' ')
            cache[sid] = text.encode('utf-8')
        except TwitterError as e:
            if e.e.code == 429:
                rate = t.application.rate_limit_status()
                reset = rate['resources']['statuses']['/statuses/show/:id']['reset']
                now = datetime.datetime.today()
                future = datetime.datetime.fromtimestamp(reset)
                seconds = (future-now).seconds+1
                if seconds < 10000:
                    sys.stderr.write("Rate limit exceeded, sleeping for %s seconds until %s\n" % (seconds, future))
                    time.sleep(seconds)
            else:
                cache[sid] = 'Not Available'.encode('utf-8')

    text = cache[sid]

    args.output.write("\t".join(fields + [text.decode("utf-8")]) + '\n')