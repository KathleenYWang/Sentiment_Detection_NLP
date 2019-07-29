# Hierarchical attention for text classification in PyTorch

This model is modified from Chenyang Huang's [repo](https://github.com/chenyangh/SemEval2019Task3). 
This hierarchical attention builds attention for each sentence in the text, and then builds another attention layer on top of all sentences for the final prediction. 

## Structure of the HA directory

```text
├── HA
|  └── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  └── model
|  |  └── ha.py　　
|  └── module　　　
|  |  └── create_data.py　
|  |  └── evaluate.py　
|  |  └── lstm_hard_sigmoid.py
|  |  └── preprocessor.py
|  |  └── self_attention.py　
|  |  └── torch_moji.py　
|  └── utils
|  |  └── early_stopping.py　
|  |  └── focalloss.py　
|  |  └── tweet_processor.py
|  └── inference_ha.py #Use this for inference
|  └── train_ha.py #Use this for training
```
## Environment Setup

PyTorch1.0 with Python 3.6 serve as the backbones of this project.

The code is using one GPU by default, you have to modify the code to make it running on CPU or multiple GPUs.

Follow the instructions here to build a conda environment that runs PyTorch1.0 and Python 3.6

```console
conda update conda                                   # base conda installation
conda update anaconda                                # packages
conda list                                           # List all packages
conda create --name pytorch anaconda python=3.6
conda activate pytorch
conda update --all
conda install pytorch torchvision cuda100 -c pytorch
```

## Other Dependencies
We use three embeddings, GloVe, ELMo and DeepEmoji, together to create embeddings for each word. 

First, download the DeepMoji pretrain model if you haven't used it before. We are using the implementation by Hugginface (https://github.com/huggingface/torchMoji).

To avoid the conficts of some packages, we used the fork from Chenyang Huang directly (https://github.com/chenyangh/torchMoji.git). Following the instructions for installation and download the model by the following script (under the direcory of their repo):

```console
git clone https://github.com/chenyangh/torchMoji.git
cd torchMoji
pip install -e .
python scripts/download_weights.py
```
We also have to download GloVe before hand. The version we have is glove.840B.300d.txt. Note the path to the GloVe and specify that in the basic_config.py file.

For ELmo, we use AllenNLP. 
```console
git clone https://github.com/allenai/allennlp.git
cd allennlp
pip install --editable .
```

## For Training
To train, first we combine all the possible training datasets and create vocabulary using load_data_context and build_vocab in the create_data.py module so that we can have a consistent vocab throughout training. Store these vocab files and remember their paths in the config file

Then, visit basic_config.py to specify:  
1.  config['output']['result']
2.  config['infer']['word2id']
3.  config['infer']['id2word']
4.  config['emb']['glove_path']
5.  config['emb']['bert_vocab_path']

The other model specifics can also be changed in the basic_config.py file such as the learning rate and batch size. 

The train data path, test data path and model save point paths have to be specified in the script to call train_ha.py.

```console
python train_ha.py -folds 2 -epoch 6 -input_path '/data/ToxicityDataOld/train.csv' \
-test_path '/data/ToxicityDataOld/test_with_labels.csv' \
-out_path '/data/SuperMod/hapy_state_wiki.pth'
```

To continue training an existing model, simply add `-cont True` at the end of calling script

## For Inference

To use an existing model for inference, call the inference_ha.py script and specify the model location. The output of inference is specified in the basic_config.py file
```console
python inference_ha.py -test_path "/data/SuperMod/final_test.csv"  -out_path '/data/SuperMod/hapy_state.pt'
```



