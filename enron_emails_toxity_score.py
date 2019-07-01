#!/usr/bin/env python
import argparse
import pandas as pd
import pickle
import time
import perspective
import os,sys,inspect, email,re
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import googleapi


PerspectiveAPI = googleapi.GOOGLEAPI

def main(PerspectiveAPI):
    parser = argparse.ArgumentParser(description='Add toxicity score to email dataframe',
                                     prog='enron_emails_toxity_score.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-f",
        "--file",
        dest='file',
        required=True,
        type=str,
        help='Input email pickle file')
    parser.add_argument("-a", "--api", dest='api', required=True,
                        type=str, help='API file')
    parser.add_argument("-o", "--out", dest='out',
                        type=str, help='output file', default='scored.pkl')
    parser.add_argument('-t', action='store_true')

    args = parser.parse_args()
    
    with (open(args.file, "rb")) as openfile:
        df = pickle.load(openfile)
        
    df = df[10000:20000]

    with open(args.api) as f:
         google_api_key = PerspectiveAPI
    client = perspective.Perspective(google_api_key)

    def calculate_score(text):
        try:
            time.sleep(1) # limit API calls to 1 per second
            score = client.score(text, tests=["TOXICITY"])
        except:
            return -1.0
        else:
            return score["TOXICITY"].score

    if (args.t):
        df = df.head(10)

    df['toxicity_score'] = df.content.apply(calculate_score)
    df.to_pickle(args.out)

if __name__ == '__main__':
    main(PerspectiveAPI)
