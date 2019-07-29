#encoding:utf-8
from os import path
import multiprocessing
from pathlib import Path
"""
"""
BASE_DIR = Path('/root/projects/SuperMod/SentimentDetectionNLP/HRLCE')

configs = {

    'task':'multi label',

    'output':{
        'checkpoint_dir': '/data/SuperMod/hapy_state.pt',
        'cache_dir': BASE_DIR / 'model/',
        'result': '/data/SuperMod/result_wiki.pkl'
    },
    
    'emb':{
        'bert_vocab_path': '/root/projects/SuperMod/SentimentDetectionNLP/bertmodel/pybert/model/pretrain/uncased_L-12_H-768_A-12/vocab.txt',
        'glove_path': '/data/glove/glove.840B.300d.txt'
        },

    'train':{
        'EMAI_PAD_LEN': 12,
        'EMOJ_SENT_PAD_LEN': 30,
        'SENT_PAD_LEN': 30,   
        'FILL_VOCAB': True,
        'BATCH_SIZE': 200,
        'CLIP': 0.888,
        'EARLY_STOP_PATIENCE': 1,
        'LAMBDA1': 0,
        'LAMBDA2': 0,
        'FLAT': 1,
        'GAMMA': 0.2,
        'RANDOM_SEED': 0,
        'NUM_OF_VOCAB': 20000,
        'learning_rate': 5e-4,
        'loss': 'ce'
    },

    
    'model':{
        'nb_tokens': 50000,
        'embed_dropout_rate': 0.2,
        'final_dropout_rate': 0.2,
        'deepmoji_dim': 2304,
        'deepmoji_out': 300,
        'SENT_EMB_DIM': 300,
        'elmo_dim': 1024,
        'layers': 2,
        'SENT_HIDDEN_SIZE': 1200,
        'CTX_LSTM_DIM': 100,
        'elmo_o': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        'elmo_w': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
         },
    
    
    
    'infer':{
        'batch_size':400,
        'word2id': '/data/SuperMod/word2id.pkl',
        'id2word': '/data/SuperMod/id2word.pkl',
        'result': '/data/SuperMod/infer_result.pkl'
        
        
    },
    
    'callbacks':{
        'lr_patience': 5, # number of epochs with no improvement after which learning rate will be reduced.
        'mode': 'min',    # one of {min, max}
        'monitor': 'valid_loss',
        'early_patience': 20,
        'save_best_only': True,
        'save_checkpoint_freq': 10
    },

}
