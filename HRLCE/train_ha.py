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
import pickle as pkl
import sys
from copy import deepcopy
import argparse
import random
from utils.focalloss import FocalLoss
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
from module import create_data


parser = argparse.ArgumentParser(description='Options')
parser.add_argument('-folds', default=9, type=int,
                    help="num of folds")
parser.add_argument('-epoch', type=int,
                    help="please specify the number of epochs to run for each fold")
parser.add_argument('-input_path', type=str,
                    help="please specify the training data input")
parser.add_argument('-test_path', type=str,
                    help="please specify the test data input")
parser.add_argument('-out_path', type=str,
                    help="please specify the path to save the model")
parser.add_argument('-cont', default=False, type=bool,
                    help="please specify if this is continued training of an existing loaded model")
opt = parser.parse_args()


def main():
    
    ##########  Set Assumptions ############
    ##########  Set Assumptions ############
    NUM_OF_FOLD = opt.folds
    MAX_EPOCH = opt.epoch
    input_path = opt.input_path
    CONTINUE = opt.cont

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
    loss = config['train']['loss']
    # fix random seeds to ensure replicability
    RANDOM_SEED = config['train']['RANDOM_SEED']
    
    # set to one fold

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
    


    # use pre-built vocab that contains both email and wiki data
    word2id_path = config['infer']['word2id']
    id2word_path = config['infer']['id2word']

    with open(word2id_path, 'rb') as w:
        word2id = pkl.load(w)
    with open(id2word_path, 'rb') as i:
        id2word = pkl.load(i)
    num_of_vocab = len(word2id)

    emb = create_data.build_embedding(id2word, GLOVE_EMB_PATH, num_of_vocab)
    
    # load data
    train_file = input_path
    data_list, target_list = create_data.load_data_context(data_path=train_file)


    # load final test data
    final_test_file = opt.test_path
    final_test_data_list, final_test_target_list = create_data.load_data_context(data_path=final_test_file)

    # convert to TestData class
    # then use Dataloader from torch.utils.data to create batches
    final_test_data_set = create_data.TestDataSet(final_test_data_list, EMAI_PAD_LEN, SENT_PAD_LEN, word2id, id2word, emoji_st, use_unk=False)
    final_test_data_loader = create_data.DataLoader(final_test_data_set, batch_size=BATCH_SIZE, shuffle=True)
    print("Size of final test data", len(final_test_data_set))


    X = data_list
    y = target_list
    y = np.array(y)

    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)

    # train dev split
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=NUM_OF_FOLD, random_state=0)

 
    # for this version, remove multiple folds
    
    real_test_results = []

    # Train one fold at a time (for cross validation)

    def one_fold(num_fold, train_index, dev_index):
        print("Training on fold:", num_fold)
        X_train, X_dev = [X[i] for i in train_index], [X[i] for i in dev_index]
        y_train, y_dev = y[train_index], y[dev_index]

        # construct data loader
        # for one fold, test data comes from k fold split.
        train_data_set = create_data.TrainDataSet(X_train, y_train, EMAI_PAD_LEN, SENT_PAD_LEN, word2id, emoji_st, use_unk=True)

        dev_data_set = create_data.TrainDataSet(X_dev, y_dev, EMAI_PAD_LEN, SENT_PAD_LEN, word2id, emoji_st, use_unk=True)
        dev_data_loader = DataLoader(dev_data_set, batch_size=BATCH_SIZE, shuffle=False)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        final_pred_best = None
        
        # This is to prevent model diverge, once happen, retrain
        while True:
                        
            is_diverged = False
            # Model is defined in HierarchicalPredictor
            
            
            if CONTINUE:            
                model = torch.load(opt.out_path)  
            else:
                model = HierarchicalAttPredictor(SENT_EMB_DIM, SENT_HIDDEN_SIZE, CTX_LSTM_DIM, num_of_vocab, SENT_PAD_LEN , id2word, USE_ELMO=True, ADD_LINEAR=False)          
                model.load_embedding(emb)
                model.deepmoji_model.load_specific_weights(PRETRAINED_PATH, exclude_names=['output_layer'])                
                        
            model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

            # loss_criterion_binary = nn.CrossEntropyLoss(weight=weight_list_binary)  #
            if loss == 'focal':
                loss_criterion = FocalLoss(gamma= opt.focal )

            elif loss == 'ce':
                loss_criterion = nn.BCELoss()

            es = EarlyStopping(patience=EARLY_STOP_PATIENCE)
            final_pred_list_test = None

            
            for num_epoch in range(MAX_EPOCH):
                
                # to ensure shuffle at ever epoch
                train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle = True)

                print('Begin training epoch:', num_epoch, end = '...\t')
                sys.stdout.flush()

                # stepping scheduler
                scheduler.step(num_epoch)
                print('Current learning rate', scheduler.get_lr())
                
                ## Training step
                train_loss = 0
                model.train()

                for i, (a, a_len, emoji_a, e_c) \
                        in tqdm(enumerate(train_data_loader), total=len(train_data_set)/BATCH_SIZE):
                    
                    optimizer.zero_grad()
                    e_c = e_c.type(torch.float)
                    pred = model(a.cuda(), a_len, emoji_a.cuda())
                    loss_label = loss_criterion(pred.squeeze(1), e_c.view(-1).cuda()).cuda()

                    # training trilogy
                    loss_label.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                    optimizer.step()

                    train_loss += loss_label.data.cpu().numpy() * a.shape[0]
                    del pred, loss_label
            

                ## Evaluatation step
                model.eval()
                dev_loss = 0
                # pred_list = []
                for i, (a, a_len, emoji_a, e_c) in enumerate(dev_data_loader):
                    
                    with torch.no_grad():
                        e_c = e_c.type(torch.float)
                        pred = model(a.cuda(), a_len, emoji_a.cuda())
                        
                        loss_label = loss_criterion(pred.squeeze(1), e_c.view(-1).cuda()).cuda()
                        
                        dev_loss += loss_label.data.cpu().numpy() * a.shape[0]

                        # pred_list.append(pred.data.cpu().numpy())
                        # gold_list.append(e_c.numpy())
                        del pred,  loss_label
                        
                print('Training loss:', train_loss / len(train_data_set), end='\t')
                print('Dev loss:', dev_loss / len(dev_data_set))
                
                # print(classification_report(gold_list, pred_list, target_names=EMOS))
                # get_metrics(pred_list, gold_list)


                # Gold Test testing
                print('Final test testing...')
                final_pred_list_test = []
                model.eval()

                for i, (a, a_len, emoji_a) in enumerate(final_test_data_loader):

                    with torch.no_grad():

                        pred = model(a.cuda(), a_len, emoji_a.cuda())

                        final_pred_list_test.append(pred.data.cpu().numpy())
                    del  a, pred
                print("final_pred_list_test",  len(final_pred_list_test) )  
                final_pred_list_test = np.concatenate(final_pred_list_test, axis=0)
                final_pred_list_test = np.squeeze(final_pred_list_test, axis=1)
                print("final_pred_list_test_concat",  len(final_pred_list_test) )  
                
                if dev_loss/len(dev_data_set) > 1.3 and num_epoch > 4:
                    print("Model diverged, retry")
                    is_diverged = True
                    break

                if es.step(dev_loss):  # overfitting
                    print('overfitting, loading best model ...')
                    break
                else:
                    if es.is_best():
                        print('saving best model ...')
                        if final_pred_best is not None:
                            del final_pred_best
                        final_pred_best = deepcopy(final_pred_list_test)
 
                    else:
                        print('not best model, ignoring ...')
                        if final_pred_best is None:
                            final_pred_best = deepcopy(final_pred_list_test)
                        

            if is_diverged:
                print("Reinitialize model ...")
                del model
                
                continue

            real_test_results.append(np.asarray(final_pred_best))
            # saving model for inference
            torch.save(model, opt.out_path)
            del model
            break

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Training the folds
    for idx, (_train_index, _dev_index) in enumerate(skf.split(X, y)):
        print('Train size:', len(_train_index), 'Dev size:', len(_dev_index))
        one_fold(idx, _train_index, _dev_index)


    real_test_results = np.asarray(real_test_results)

    # since we only have 1 value per row per fold, just average across folds for the final value
    mj = np.mean(real_test_results, axis=0)  

    ### This is loading final test results to get metric
    print('Gold TESTING RESULTS')
    print(np.asarray(final_test_target_list).shape)
    print(np.asarray(mj).shape)    
    get_metrics(np.asarray(final_test_target_list), np.asarray(mj))
    



main()
