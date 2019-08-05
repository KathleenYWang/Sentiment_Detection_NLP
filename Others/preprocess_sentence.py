from __future__ import unicode_literals, print_function
from nltk.tokenize import sent_tokenize
import pandas as pd
from spacy.lang.en import English
import copy 
import numpy as np

def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email


def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from')
    }

def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results

def break_into_lines (email):

    try:
        doc = sent_tokenize(email)
        return doc
    except:
        return []
def break_into_lines_spacy (email):
    
    nlp = English ()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    try:
        doc = nlp (email)
        return list(doc.sents)
    except:
        return []

def further_break_long_sentences (email_sent_list):
    return_list = []
    for ind, sent in enumerate (email_sent_list):
        if len(sent) < 50:
            return_list.extend(sent)
        else:
            for i in range(len(sent) // 50 + 1):
                start = i * 50
                end = min( (1 + i) * 50, len(sent))
                return_list.extend(sent[start, end])             
    return return_list

def avg_sent_len_f (sent):
    if sent == []:
        return 0
    else:
        return np.mean([len(s) for s in sent])

# def parse_df (email_df_in):

datadir = "/data/SuperMod/emails.csv"
enrondata = pd.read_csv(datadir)
email_df = pd.DataFrame(parse_into_emails(enrondata.message))

email_body = email_df.body
email_sentence = email_body.map(break_into_lines)
email_forward_cnt = email_body.map(lambda x: len(x.split("---------------------- Forwarded by "))-1)
email_reply_cnt = email_body.map(lambda x: len(x.split("-----Original Message"))-1)
# new_message_only = email_body.map(lambda x: x.split("---------------------- Forwarded by ")[0].split("-----Original Message")[0].split("--------- Inline attachment follows")[0] )

# new_message_sentence = new_message_only.map(break_into_lines)
email_sentence_cnt = email_sentence.map(lambda x: len(x))
# new_message_sentence_cnt = new_message_sentence.map(lambda x: len(x))
avg_sent_len = email_sentence.map(avg_sent_len_f)
# avg_sent_len_newe = new_message_sentence.map(avg_sent_len_f)

np.save("/data/SuperMod/email_sentence", email_sentence)
np.save("/data/SuperMod/email_forward_cnt", email_forward_cnt)
np.save("/data/SuperMod/email_reply_cnt", email_reply_cnt)
# np.save("/data/SuperMod/new_message_only", new_message_only)
# np.save("/data/SuperMod/new_message_sentence", new_message_sentence)
np.save("/data/SuperMod/email_sentence_cnt", email_sentence_cnt)
# np.save("/data/SuperMod/new_message_sentence_cnt", new_message_sentence_cnt)
np.save("/data/SuperMod/avg_sent_len", avg_sent_len)
# np.save("/data/SuperMod/avg_sent_len_newe", avg_sent_len_newe)







# with open('/data/SuperMod/full_email.csv', 'a') as f:
#     parsed.to_csv(f, header=False)


# # for i in range(total_len//500 +1):
# #     start = i * 500
# #     end = min(total_len, (1+i)*500)
# #     parsed = parse_df(email_df.iloc[start:end,])
# #     with open('/data/SuperMod/full_email.csv', 'a') as f:
# #         parsed.to_csv(f, header=False)