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

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('-folds', default=9, type=int,
                    help="num of folds")
parser.add_argument('-bs', default=2, type=int,
                    help="batch size")
parser.add_argument('-postname', default='', type=str,
                    help="name that will be added at the end of generated file")
parser.add_argument('-gamma', default=0.2, type=float,
                    help="the decay of the ")
parser.add_argument('-lr', default=5e-4, type=float,
                    help="learning rate")
parser.add_argument('-lbd1', default=0, type=float,
                    help="lambda1 is for MTL")
parser.add_argument('-lbd2', default=0, type=float,
                    help="lambda2 is for optimizing only the emotional labels")
parser.add_argument('-patience', default=1, type=int,
                    help="patience of early stopping")
parser.add_argument('-flat', default=1, type=float,
                    help="flatten para")
parser.add_argument('-focal', default=2, type=int,
                    help="gamma value for focal loss, default 2 ")
parser.add_argument('-w', default=2, type=int,
                    help="type of weight ")
parser.add_argument('-loss', default='ce', type=str,
                    help="ce or focal ")
parser.add_argument('-dims', default=1500, type=int,
                    help="sent hidden size")
parser.add_argument('-dime', default=100, type=int,
                    help="email context hidden size")
parser.add_argument('-glovepath', type=str,
                    help="please specify the path to a GloVe 300d emb file")
parser.add_argument('-epoch', type=int,
                    help="please specify the number of epochs to run for each fold")
parser.add_argument('-input_path', type=str,
                    help="please specify the training data input")
parser.add_argument('-test_path', type=str,
                    help="please specify the test data input")
parser.add_argument('-out_path', type=str,
                    help="please specify the path to save the model")

opt = parser.parse_args()

##########  Set Assumptions ############
##########  Set Assumptions ############
NUM_OF_FOLD = opt.folds
learning_rate = opt.lr
MAX_EPOCH = opt.epoch

EMAI_PAD_LEN = 10
EMOJ_SENT_PAD_LEN = SENT_PAD_LEN = 30
SENT_EMB_DIM = 300

FILL_VOCAB = True
BATCH_SIZE = opt.bs

SENT_HIDDEN_SIZE = opt.dims
CTX_LSTM_DIM = opt.dime

CLIP = 0.888
EARLY_STOP_PATIENCE = opt.patience
LAMBDA1 = opt.lbd1
LAMBDA2 = opt.lbd2
FLAT = opt.flat
GAMMA = opt.gamma
# fix random seeds to ensure replicability
RANDOM_SEED = 0
NUM_OF_VOCAB = 5000
input_path = opt.input_path


torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

GLOVE_EMB_PATH = opt.glovepath

preprocessor = EnglishPreProcessor()
tokenizer = BertTokenizer(vocab_file='/root/projects/SuperMod/SentimentDetectionNLP/bertmodel/pybert/model/pretrain/uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)

word2id_path = config['infer']['word2id']
id2word_path = config['infer']['id2word']


print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
emoji_st = SentenceTokenizer(vocabulary, EMOJ_SENT_PAD_LEN)

result_path = config['output']['result']

def load_data_context(data_path='/data/SuperMod/test_data.txt', is_train=True):

    data_list = []
    target_list = []
    
    df = pd.read_csv(data_path, encoding="utf8")
    
    if len(df.columns) > 4:
        data_list = df.comment_text.tolist()
        target_list = df.toxic.tolist()        
    else:
        data_list = df.comment_text.tolist()
        target_list = df.toxicity.tolist()
    
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


def main():
    num_of_vocab = NUM_OF_VOCAB

    # load data
    train_file = input_path
    data_list, target_list = load_data_context(data_path=train_file)

    # dev set
    dev_file = input_path
    dev_data_list, dev_target_list = load_data_context(data_path=dev_file)

    # load final test data
    final_test_file = opt.test_path
    final_test_data_list, final_test_target_list = load_data_context(data_path=final_test_file)

    # build vocab (load vocab from the larger vocab pool)
#     word2id, id2word, num_of_vocab = build_vocab([data_list, dev_data_list, final_test_data_list], num_of_vocab,
#                                                  FILL_VOCAB)

    
    with open(word2id_path, 'rb') as w:
        word2id = pkl.load(w)
    with open(id2word_path, 'rb') as i:
        id2word = pkl.load(i)
    num_of_vocab = len(word2id)

        
    emb = build_embedding(id2word, GLOVE_EMB_PATH, num_of_vocab)    
    
## we don''t really have multiple test sets        
#     test_data_set = TrainDataSet(dev_data_list, dev_target_list, EMAI_PAD_LEN, SENT_PAD_LEN, word2id, use_unk=True)
#     test_data_loader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False)
#     print("Size of test data", len(test_data_set))
    # ex_id2word, unk_words_idx = test_data_set.get_ex_id2word_unk_words()

    # convert to TestData class
    # then use Dataloader from torch.utils.data to create batches
    final_test_data_set = TestDataSet(final_test_data_list, EMAI_PAD_LEN, SENT_PAD_LEN, word2id,id2word, use_unk=False)
    final_test_data_loader = DataLoader(final_test_data_set, batch_size=BATCH_SIZE, shuffle=False)
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

    real_test_results = []
    
    # Train one fold at a time (for cross validation)

    def one_fold(num_fold, train_index, dev_index):
        print("Training on fold:", num_fold)
        X_train, X_dev = [X[i] for i in train_index], [X[i] for i in dev_index]
        y_train, y_dev = y[train_index], y[dev_index]

        # construct data loader
        # for one fold, test data comes from k fold split.
        train_data_set = TrainDataSet(X_train, y_train, EMAI_PAD_LEN, SENT_PAD_LEN, word2id, use_unk=True)

        dev_data_set = TrainDataSet(X_dev, y_dev, EMAI_PAD_LEN, SENT_PAD_LEN, word2id, use_unk=True)
        dev_data_loader = DataLoader(dev_data_set, batch_size=BATCH_SIZE, shuffle=False)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        final_pred_best = None
        
        # This is to prevent model diverge, once happen, retrain
        while True:
            is_diverged = False
            # Model is defined in HierarchicalPredictor
            model = HierarchicalAttPredictor(SENT_EMB_DIM, SENT_HIDDEN_SIZE, CTX_LSTM_DIM, num_of_vocab, SENT_PAD_LEN , id2word, USE_ELMO=True, ADD_LINEAR=False) 
            model.load_embedding(emb)
            model.deepmoji_model.load_specific_weights(PRETRAINED_PATH, exclude_names=['output_layer'])
            model.cuda()

            # model = nn.DataParallel(model)
            # model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True) #
            # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

            # loss_criterion_binary = nn.CrossEntropyLoss(weight=weight_list_binary)  #
            if opt.loss == 'focal':
                loss_criterion = FocalLoss(gamma= opt.focal )

            elif opt.loss == 'ce':
                loss_criterion = nn.BCELoss()



            es = EarlyStopping(patience=EARLY_STOP_PATIENCE)
            final_pred_list_test = None

            result_print = {}
            
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
                
                
                accuracy, precision, recall, f1 = get_metrics(np.asarray(final_test_target_list), np.asarray(final_pred_list_test))
                
                result_print.update( {num_epoch: [ accuracy, precision, recall, f1]} )
    
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
            
                           
                
            with open(result_path, 'wb') as w:
                pkl.dump(result_print, w)

            if is_diverged:
                print("Reinitialize model ...")
                del model
                
                continue

            real_test_results.append(np.asarray(final_pred_best))
            # saving model for inference
            torch.save(model.state_dict(), opt.out_path)
            del model
            break

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Training the folds
    for idx, (_train_index, _dev_index) in enumerate(skf.split(X, y)):
        print('Train size:', len(_train_index), 'Dev size:', len(_dev_index))
        one_fold(idx, _train_index, _dev_index)

#     # Function of majority voting
#     # Need this to vote across different folds
#     def find_majority(k):
#         myMap = {}
#         maximum = ('', 0)  # (occurring element, occurrences)
#         for n in k:
#             if n in myMap:
#                 myMap[n] += 1
#             else:
#                 myMap[n] = 1

#             # Keep track of maximum on the go
#             if myMap[n] > maximum[1]: maximum = (n, myMap[n])

#         return maximum


    real_test_results = np.asarray(real_test_results)

    # since we only have 1 value per row per fold, just average across folds for the final value
    mj = np.mean(real_test_results, axis=0)  
#     for col_num in range(real_test_results.shape[1]):
#         a_mj = find_majority(real_test_results[:, col_num])
#         mj.append(a_mj[0])

    ### This is loading final test results to get metric
    print('Gold TESTING RESULTS') 
    get_metrics(np.asarray(final_test_target_list), np.asarray(mj))
    



main()
