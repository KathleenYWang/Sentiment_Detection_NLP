#encoding:utf-8
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
from module.preprocessor import EnglishPreProcessor
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.tweet_processor import processing_pipeline
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import copy
from tqdm import tqdm
import numpy as np
import torch
import pickle as pkl
from config.basic_config import configs as config
import json
from torch.utils.data import Dataset, DataLoader


BERT_EMB_PATH = config['emb']['bert_vocab_path']
GLOVE_EMB_PATH = config['emb']['glove_path']
preprocessor = EnglishPreProcessor()
tokenizer = BertTokenizer(vocab_file=BERT_EMB_PATH, do_lower_case=True)
EMOJ_SENT_PAD_LEN = config['train']['EMOJ_SENT_PAD_LEN']


print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
emoji_st = SentenceTokenizer(vocabulary, EMOJ_SENT_PAD_LEN)


def load_data_context(data_path='/data/SuperMod/test_data.txt', is_train=True):

    data_list = []
    target_list = []
    
    df = pd.read_csv(data_path)
    
    if len(df.columns) > 3:
        data_list = df.comment_text.tolist()
        target_list = df.toxic.tolist()        
    else:
        data_list = df.content.tolist()
        target_list = df.supermod.tolist()
    
    clean_sent_list = [sent_tokenize(processing_pipeline(email)) for email in data_list]

    if is_train:
        return clean_sent_list, target_list
    else:
        return clean_sent_list

def clean_sentences(sent_text):
    """
    This function detects if a line should be removed (all empty or no words)
    And returns a cleaned sentences without duplicated tokens
    """
    to_keep = False
    if re.match(".*[a-zA-Z]+.*", sent_text):
        to_keep = True
    remove_dup = re.sub(r'(.+?)\1+', r'\1', processing_pipeline(sent_text))
    return to_keep, remove_dup
    

def build_vocab(clean_sent_lists, vocab_size, fill_vocab=False):
    """
    get all the words from the list, and sort them by count
    Then convert them into ids.
    This is simply word split, can improve with other tokenizer
    clean_sent_lists is a list of list of tokenized emails
    """
    word_count = {}
    word2id = {}
    id2word = {}
    
    data_list_list = []
    
    for data_list in clean_sent_lists:
        data_list_list.extend(data_list)
    
    for emails in data_list_list:
        for sentence in emails:
            keep, sents = clean_sentences(sentence)
            if keep:
                for word in tokenizer.tokenize(sents):
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1 

    word_list = [x for x, _ in sorted(word_count.items(), key=lambda v: v[1], reverse=True)]
    print('found', len(word_count), 'words')

    if len(word_count) < vocab_size:
        raise Exception('Vocab less than requested!!!')

    # add <pad> first
    word2id['<pad>'] = 0
    id2word[0] = '<pad>'

    word2id['<unk>'] = 1
    id2word[1] = '<unk>'
    word2id['<empty>'] = 2
    id2word[2] = '<empty>'

    n = len(word2id)
    if not fill_vocab:
        word_list = word_list[:vocab_size - n]

    for word in word_list:
        word2id[word] = n
        id2word[n] = word
        n += 1

    if fill_vocab:
        print('filling vocab to', len(id2word))
        return word2id, id2word, len(id2word)
    return word2id, id2word, len(word2id)



class TrainDataSet(Dataset):
    def __init__(self, data_list, target_list, emai_pad_len, sent_pad_len, word2id,  use_unk=False):

        self.sent_pad_len = sent_pad_len
        self.emai_pad_len = emai_pad_len
        self.word2id = word2id
        self.pad_int = self.word2id['<pad>']

        self.use_unk = use_unk

        # internal data
        # only have one input, not 3
        self.a = []
        self.a_len = []
        self.emoji_a = []
        
        # e_c is the label of emotions
        # for our project, the label is already 0, 1
        # so remove binary and label translation
        self.e_c = []
#         self.e_c_binary = []
#         self.e_c_emo = []
        self.num_empty_lines = 0

        self.weights = []
        # prepare dataset
        self.read_data(data_list, target_list)
        
    def sent_to_ids(self, sent_text):
        """
        convert words into ids, 
        this takes in sentences, not emails
        Then tokenize and convert to ids for each sentence
        """        
        tokens = tokenizer.tokenize(sent_text)
        
        if self.use_unk:
            tmp = [self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens]
        else:
            tmp = [self.word2id[x] for x in tokens if x in self.word2id]
        if len(tmp) == 0:
            tmp = [self.word2id['<empty>']]
            self.num_empty_lines += 1

        # PADDING
        if len(tmp) > self.sent_pad_len:
            tmp = tmp[: self.sent_pad_len]
        text_len = max(len(tmp), 1)

        tmp = tmp + [self.pad_int] * (self.sent_pad_len - len(tmp))
        # no need to pad emoji
        a_emoji = emoji_st.tokenize_sentences([sent_text])[0].reshape((-1)).astype(np.int64)

        return tmp, text_len, a_emoji            

    def email_to_ids(self, email_text):
        """
        convert words into ids, 
        first split email into list of sentences, then use sent_to_ids
        Then tokenize and convert to ids for each sentence
        """ 
        tmp = []
        tmp_len = []
        a_emoji = []
        
        for sents in email_text:
            keep, sents = clean_sentences(sents)
            if keep:                
                a, a_len, emoji_sent = self.sent_to_ids(sents)
                tmp.append(a)
                tmp_len.append(a_len)
                a_emoji.append(emoji_sent)
            
        if len(email_text) > self.emai_pad_len:
            tmp = tmp[: self.emai_pad_len]
            tmp_len = tmp_len[: self.emai_pad_len]
            a_emoji = a_emoji[: self.emai_pad_len]
        
        fill_times = self.emai_pad_len - len(tmp)          
        # have to make sure the fillers look accurate for it to work
        tmp = tmp +  [[self.pad_int] * self.sent_pad_len for i in range(fill_times)]      
        tmp_len = tmp_len + [1] *  fill_times    
        a_emoji = a_emoji + [[self.pad_int]  * EMOJ_SENT_PAD_LEN for i in range(fill_times)]  
 
        return tmp, tmp_len, a_emoji

    def read_data(self, data_list, target_list):
        """
        data_list: contains both cleaned and raw data
        """
        assert len(data_list) == len(target_list)

        for X, y in zip(data_list, target_list):
            
            # convert clean sentence to ids
            a, a_len, a_emoji = self.email_to_ids(X)

            self.a.append(a)
            self.a_len.append(a_len)            
            self.emoji_a.append(a_emoji)

            # append the target
            self.e_c.append(int(y)) 
    
    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return  torch.LongTensor(self.a[idx]), \
            torch.LongTensor(self.a_len[idx]),\
            torch.LongTensor(self.emoji_a[idx]),\
            torch.LongTensor([self.e_c[idx]])
    

class TestDataSet(Dataset):
    """
    Process test dataset, the difference is this does not read labels
    """
    
    def __init__(self, data_list,  emai_pad_len, sent_pad_len,  word2id, id2word, use_unk=False):

        self.sent_pad_len = sent_pad_len
        self.emai_pad_len = emai_pad_len
        self.word2id = word2id
        self.pad_int = word2id['<pad>']

        self.use_unk = use_unk

        # internal data
        self.a = []
        self.a_len = []
        self.emoji_a = []


        self.num_empty_lines = 0
        # prepare dataset
        self.ex_word2id = copy.deepcopy(word2id)
        self.ex_id2word = copy.deepcopy(id2word)
        self.unk_words_idx = set()
        self.read_data(data_list)
   
        
    def sent_to_ids(self, sent_text):
        """
        convert words into ids, 
        this takes in sentences, not emails
        Then tokenize and convert to ids for each sentence
        """        
        tokens = tokenizer.tokenize(sent_text)
        
        
        if self.use_unk:
            tmp = [self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens]
        else:
            tmp = [self.word2id[x] for x in tokens if x in self.word2id]
        if len(tmp) == 0:
            tmp = [self.word2id['<empty>']]
            self.num_empty_lines += 1

        # PADDING
        if len(tmp) > self.sent_pad_len:
            tmp = tmp[: self.sent_pad_len]
        text_len = max(len(tmp),1)

        tmp = tmp + [self.pad_int] * (self.sent_pad_len - len(tmp))
        # no need to pad emoji
        a_emoji = emoji_st.tokenize_sentences([sent_text])[0].reshape((-1)).astype(np.int64)

        return tmp, text_len, a_emoji
        

    def email_to_ids(self, email_text):
        """
        convert words into ids, 
        first split email into list of sentences, then use sent_to_ids
        Then tokenize and convert to ids for each sentence
        """ 
        tmp = []
        tmp_len = []
        a_emoji = []
                
        for sents in email_text:
            keep, sents = clean_sentences(sents)
            if keep:                
                a, a_len, emoji_sent = self.sent_to_ids(sents)
                tmp.append(a)
                tmp_len.append(a_len)
                a_emoji.append(emoji_sent)
            
        if len(email_text) > self.emai_pad_len:
            tmp = tmp[: self.emai_pad_len]
            tmp_len = tmp_len[: self.emai_pad_len]
            a_emoji = a_emoji[: self.emai_pad_len]
            
            
        fill_times = self.emai_pad_len - len(tmp)    
        
        tmp = tmp +  [[self.pad_int] * self.sent_pad_len for i in range(fill_times)]      
        tmp_len = tmp_len + [1] *  fill_times    
        a_emoji = a_emoji + [[self.pad_int]  * EMOJ_SENT_PAD_LEN for i in range(fill_times)]  
 
        return tmp, tmp_len, a_emoji


    def read_data(self, data_list):
        """
        data_list: originally contains both cleaned and raw data
        since raw data not used, get rid of it
        """        
        for X in data_list:

            a, a_len, a_emoji = self.email_to_ids(X)

            self.a.append(a)
            self.a_len.append(a_len)            
            self.emoji_a.append(a_emoji)
            

        
    def __len__(self):
        return len(self.a) 

    def __getitem__(self, idx):
        return torch.LongTensor(self.a[idx]),\
               torch.LongTensor(self.a_len[idx]), \
               torch.LongTensor(self.emoji_a[idx])
  

    
def build_embedding(id2word, fname, num_of_vocab):
    """
    Build Glove Embedding, fname is the glove embedding path
    """
    import io

    def load_vectors(fname):
        print("Loading Glove Model")
        f = open(fname, 'r', encoding='utf8')
        model = {}
        for line in tqdm(f.readlines(), total=2196017):
            values = line.split(' ')
            word = values[0]
            try:
                embedding = np.array(values[1:], dtype=np.float32)
                model[word] = embedding
            except ValueError:
                print(len(values), values[0])

        print("Done.", len(model), " words loaded!")
        f.close()
        return model

    def get_emb(emb_dict, vocab_size, embedding_dim):

        all_embs = np.stack(emb_dict.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        emb = np.random.normal(emb_mean, emb_std, (vocab_size, embedding_dim))

        num_found = 0
        print('loading glove')
        for idx in tqdm(range(vocab_size)):
            word = id2word[idx]
            if word == '<pad>' or word == '<unk>':
                emb[idx] = np.zeros([embedding_dim])
            elif word in emb_dict:
                emb[idx] = emb_dict[word]
                num_found += 1

        return emb, num_found

    pkl_path = fname + '.pkl'
    if not os.path.isfile(pkl_path):
        print('creating pkl file for the emb text file')
        emb_dict = load_vectors(fname)
        with open(pkl_path, 'wb') as f:
            pkl.dump(emb_dict, f)
    else:
        print('loading pkl file')
        with open(pkl_path, 'rb') as f:
            emb_dict = pkl.load(f)
        print('loading finished')

    emb, num_found = get_emb(emb_dict, num_of_vocab, SENT_EMB_DIM)

    print(num_found, 'of', num_of_vocab, 'found', 'coverage', num_found/num_of_vocab)

    return emb
