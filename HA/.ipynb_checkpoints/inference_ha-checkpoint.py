import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.early_stopping import EarlyStopping
import numpy as np
import copy
from tqdm import tqdm
from model.ha import HierarchicalAttPredictor
from sklearn.metrics import classification_report
from module.evaluate import load_dev_labels, get_metrics
from module import create_data
import pickle as pkl
import sys
from copy import deepcopy
import argparse
import random
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from utils.tweet_processor import processing_pipeline
import json
import pandas as pd
from nltk.tokenize import sent_tokenize
from module.preprocessor import EnglishPreProcessor
from pytorch_pretrained_bert.tokenization import BertTokenizer
from config.basic_config import configs as config

import re

parser = argparse.ArgumentParser(description='Options')

parser.add_argument('-test_path', type=str,
                    help="please specify the test data input")
parser.add_argument('-out_path', type=str,
                    help="please specify the path to save the model")

opt = parser.parse_args()


def main():
    
    ##########  Set Assumptions ############
    ##########  Set Assumptions ############
    EMAI_PAD_LEN = config['train']['EMAI_PAD_LEN']
    EMOJ_SENT_PAD_LEN = config['train']['EMOJ_SENT_PAD_LEN']
    SENT_PAD_LEN = config['train']['SENT_PAD_LEN']
    SENT_EMB_DIM = config['model']['SENT_EMB_DIM']
    learning_rate = config['train']['learning_rate']
    FILL_VOCAB = config['train']['FILL_VOCAB']
    BATCH_SIZE = config['train']['BATCH_SIZE']

    SENT_HIDDEN_SIZE = config['model']['SENT_HIDDEN_SIZE']
    CTX_LSTM_DIM = config['model']['CTX_LSTM_DIM']

    CLIP = config['train']['CLIP']
    EARLY_STOP_PATIENCE = config['train']['EARLY_STOP_PATIENCE']
    LAMBDA1 = config['train']['LAMBDA1']
    LAMBDA2 = config['train']['LAMBDA2']
    FLAT = config['train']['FLAT']
    GAMMA = config['train']['GAMMA']
    # fix random seeds to ensure replicability
    RANDOM_SEED = config['train']['RANDOM_SEED']
    NUM_OF_VOCAB = config['train']['NUM_OF_VOCAB']

    GLOVE_EMB_PATH = config['emb']['glove_path']
    bert_vocab_path = config['emb']['bert_vocab_path']

    result_path = config['output']['result']
    val_result_path =  config['infer']['result']

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
  
    preprocessor = EnglishPreProcessor()
    tokenizer = BertTokenizer(vocab_file=bert_vocab_path, do_lower_case=True)

    print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    emoji_st = SentenceTokenizer(vocabulary, EMOJ_SENT_PAD_LEN)
    
    

    word2id_path = config['infer']['word2id']
    id2word_path = config['infer']['id2word']

    with open(word2id_path, 'rb') as w:
        word2id = pkl.load(w)
    with open(id2word_path, 'rb') as i:
        id2word = pkl.load(i)
    num_of_vocab = len(word2id)
    
    emb = create_data.build_embedding(id2word, GLOVE_EMB_PATH, num_of_vocab)

    final_test_file = opt.test_path
    final_test_data_list = create_data.load_data_context(data_path=final_test_file, is_train=False)

    # load vocab
    


    final_test_data_set = create_data.TestDataSet(final_test_data_list, EMAI_PAD_LEN, SENT_PAD_LEN, word2id, id2word, emoji_st, use_unk=False)
    final_test_data_loader = DataLoader(final_test_data_set, batch_size= BATCH_SIZE, shuffle=False)

    final_pred_list_test = []
    
    # Train one fold at a time (for cross validation)
#     model = HierarchicalAttPredictor(SENT_EMB_DIM, SENT_HIDDEN_SIZE, CTX_LSTM_DIM, num_of_vocab, SENT_PAD_LEN , id2word, USE_ELMO=True, ADD_LINEAR=False) 
#     model.load_embedding(emb)
#     model.deepmoji_model.load_specific_weights(PRETRAINED_PATH, exclude_names=['output_layer'])  

    
    model = torch.load(opt.out_path) 
#     model.load_state_dict(torch.load(opt.out_path))
    model.cuda()
    model.eval()  

    for i, (a, a_len, emoji_a) in tqdm(enumerate(final_test_data_loader), total=len(final_test_data_set)/BATCH_SIZE):

        with torch.no_grad():

            pred = model(a.cuda(), a_len, emoji_a.cuda())

            final_pred_list_test.append(pred.data.cpu().numpy())

    final_pred_list_test = np.concatenate(final_pred_list_test, axis=0)
    final_pred_list_test = np.squeeze(final_pred_list_test, axis=1)
         
    with open(val_result_path, 'wb') as w:
        pkl.dump(final_pred_list_test, w, pkl.HIGHEST_PROTOCOL) 

main()
