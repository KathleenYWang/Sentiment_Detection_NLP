## This script gets the weights from sentences to detect toxic sentences

import torch
import pandas as pd
import pickle as pkl
from module.evaluate import load_dev_labels, get_metrics
import numpy as np
import argparse
import torch
from model.highlight_ha import HierarchicalAttPredictorHL
from config.basic_config import configs as config
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import random
from module.preprocessor import EnglishPreProcessor
from pytorch_pretrained_bert.tokenization import BertTokenizer
import json
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torch.utils.data import Dataset, DataLoader
from module import create_data

parser = argparse.ArgumentParser(description='Options')

parser.add_argument('-test_path', type=str,
                    help="please specify the test data input")
parser.add_argument('-out_path', type=str,
                    help="please specify the path to the saved model")

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


    model = HierarchicalAttPredictorHL(SENT_EMB_DIM, SENT_HIDDEN_SIZE, CTX_LSTM_DIM, num_of_vocab, SENT_PAD_LEN , id2word, USE_ELMO=True, ADD_LINEAR=False) 
    model.load_embedding(emb)
    model.deepmoji_model.load_specific_weights(PRETRAINED_PATH, exclude_names=['output_layer'])  
    full_model = torch.load(opt.out_path)
    # model.load_state_dict(full_model.state_dict)  
    model.cuda()
    model.eval()
    
    
    highlight_test_file = opt.test_path
    highlight_test_data_list = create_data.load_data_context(data_path=highlight_test_file, is_train=False)

    highlight_test_data_set = create_data.TestDataSet(highlight_test_data_list, EMAI_PAD_LEN, SENT_PAD_LEN, word2id, id2word, emoji_st, use_unk=False)
    highlight_test_data_loader = DataLoader(highlight_test_data_set, batch_size= BATCH_SIZE, shuffle=False)


    highlight_pred_list_test = []
    highlight_pred_weights = []

    model.cuda()
    for i, (a, a_len, emoji_a) in enumerate(highlight_test_data_loader):


            with torch.no_grad():

                out, weight = model(a.cuda(), a_len, emoji_a.cuda())            

                highlight_pred_weights.append(weight.cpu().numpy())

                highlight_pred_list_test.append(out.cpu().numpy())

            
            

    def print_sent_and_weights_hl (ind_of_sample):
    
        word_list = highlight_test_data_list[ind_of_sample]
        weight_list = highlight_pred_weights[0][ind_of_sample]

        if len(word_list) > 10:
            word_list = word_list[:10]
        elif len(word_list) < 10:
            weight_list = weight_list[:len(word_list)]
            sum_new_weights = sum(weight_list)
            weight_list = weight_list/sum(weight_list)

        weight_list = np.asarray(weight_list)
        word_list =  np.asarray(word_list)

        tox_ind = np.argsort(weight_list)

        rank = [np.where(tox_ind ==  i)[0][0] for i in range(len(tox_ind))]

        for i, (words,  score, r) in enumerate(zip(word_list, weight_list, rank )):

            print(r, words)
            
            
    for sample_id in range(len(highlight_test_data_list)):
        print("\nsample", sample_id + 1,)
        
        print_sent_and_weights_hl(sample_id)
            
            
main()
    
