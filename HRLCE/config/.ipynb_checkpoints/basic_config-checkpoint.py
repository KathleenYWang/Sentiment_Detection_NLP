#encoding:utf-8
from os import path
import multiprocessing
from pathlib import Path
"""
"""
BASE_DIR = Path('/root/projects/SuperMod/SentimentDetectionNLP/HRLCE')

configs = {

    'task':'multi label',
#     'data':{
#         'raw_data_path': BASE_DIR / 'dataset/raw/train.csv',
#         'train_file_path': BASE_DIR / 'dataset/processed/train.tsv',
#         'valid_file_path': BASE_DIR / 'dataset/processed/valid.tsv',
#         'test_file_path': BASE_DIR / 'dataset/raw/test.csv'
#     },
    'output':{
        'log_dir': BASE_DIR / 'output/log',
        'writer_dir': BASE_DIR / "output/TSboard",
        'figure_dir': BASE_DIR / "output/figure",
        'checkpoint_dir': BASE_DIR / "output/checkpoints",
        'cache_dir': BASE_DIR / 'model/',
        'result': BASE_DIR / "output/result",
    },
    'pretrained':{
        "bert":{
            'vocab_path': '/root/projects/SuperMod/SentimentDetectionNLP/bertmodel/pybert/model/pretrain/uncased_L-12_H-768_A-12/vocab.txt',
#             'tf_checkpoint_path': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/bert_model.ckpt',
#             'bert_config_file': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/bert_config.json',
#             'pytorch_model_path': BASE_DIR / 'model/pretrain/pytorch_pretrain/pytorch_model.bin',
#             'bert_model_dir': BASE_DIR / 'model/pretrain/pytorch_pretrain',
        },
        'embedding':{}
    },
    'train':{
        'EMAI_PAD_LEN': 10,
        'EMOJ_SENT_PAD_LEN': 30,
        'SENT_PAD_LEN': 30,   
        'FILL_VOCAB': True,
        'BATCH_SIZE': 128,
        'CLIP': 0.888,
        'EARLY_STOP_PATIENCE': 1,
        'LAMBDA1': 0,
        'LAMBDA2': 0,
        'FLAT': 1,
        'GAMMA': 0.2,
        'RANDOM_SEED': 0
        'NUM_OF_VOCAB': 5000
        'learning_rate': 5e-4
        
        
        'batch_size': 24,#24 on GPU 2 x NVIDIA Tesla K80 or P100   #8 with CPU with 16 GB RAM
        'epochs': 6,  # number of epochs to train
        'start_epoch': 1,
        'warmup_proportion': 0.1,# Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
        'gradient_accumulation_steps': 1,# Number of updates steps to accumulate before performing a backward/update pass.
        'learning_rate': 2e-5,
        'n_gpu': [1,0],
        'num_workers': multiprocessing.cpu_count(),
        'weight_decay': 1e-5,
        'seed':2018,
        'resume':False,
        
    

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
        
        
        
    },
    
        self.deepmoji_model = TorchMoji(nb_classes=None,
                                        nb_tokens=50000,
                                        embed_dropout_rate=0.2,
                                        final_dropout_rate=0.2)
        self.deepmoji_dim = 2304
        self.deepmoji_out = 300
        self.deepmoji2linear = nn.Linear(self.deepmoji_dim, self.deepmoji_out)

        self.elmo_dim = 1024

        self.num_layers = 2
        self.use_elmo = USE_ELMO
        if not self.use_elmo:
            self.elmo_dim = 0    
    
    'predict':{
        'batch_size':400
    },
    'callbacks':{
        'lr_patience': 5, # number of epochs with no improvement after which learning rate will be reduced.
        'mode': 'min',    # one of {min, max}
        'monitor': 'valid_loss',
        'early_patience': 20,
        'save_best_only': True,
        'save_checkpoint_freq': 10
    },
    'label2id' : {
        "toxic": 0
    },
    'model':{
        'arch':'bert'
    }
}
