import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from module.self_attention import BertAttention, AttentionOneParaPerChan
from module.torch_moji import TorchMoji
from allennlp.modules.elmo import Elmo, batch_to_ids
import pickle as pkl
import os
from tqdm import tqdm
from allennlp.commands.elmo import ElmoEmbedder
from config.basic_config import configs as config


"""
pred = model(a.cuda(), a_len, emoji_a.cuda(), elmo_a)
model = HierarchicalPredictor(SENT_EMB_DIM, SENT_HIDDEN_SIZE, num_of_vocab, USE_ELMO=True, ADD_LINEAR=False)
The input to this model has to be changed to a list of sentences as email, 

"""

options_file = config['model']['elmo_o']
weight_file = config['model']['elmo_w']
nb_tokens = config['model']['nb_tokens']
embed_dropout_rate = config['model']['embed_dropout_rate']
final_dropout_rate = config['model']['final_dropout_rate']

deepmoji_dim = config['model']['deepmoji_dim']
deepmoji_out = config['model']['deepmoji_out']
elmo_dim = config['model']['elmo_dim']
layers = config['model']['layers']

class HierarchicalAttPredictor(nn.Module):
    """
    A Hierarchical attention taking an email with multiple sentences
    """
    def __init__(self, embedding_dim, hidden_dim_s, hidden_dim_e , vocab_size, SENT_PAD_LEN, id2word, USE_ELMO, ADD_LINEAR):
        super(HierarchicalAttPredictor, self).__init__()
        self.SENT_LSTM_DIM = hidden_dim_s
        self.bidirectional = True
        self.add_linear = ADD_LINEAR

        self.sent_lstm_directions = 2 if self.bidirectional else 1
        # centence level lstm attention
        self.cent_lstm_att_fn = AttentionOneParaPerChan
        # context lstm attention
        self.ctx_lstm__att_fn = AttentionOneParaPerChan

        self.deepmoji_model = TorchMoji(nb_classes=None,
                                        nb_tokens = nb_tokens,
                                        embed_dropout_rate = embed_dropout_rate,
                                        final_dropout_rate = final_dropout_rate)
        self.deepmoji_dim = deepmoji_dim
        self.deepmoji_out = deepmoji_out
        self.deepmoji2linear = nn.Linear(self.deepmoji_dim, self.deepmoji_out)

        self.elmo_dim = elmo_dim

        self.num_layers = layers
        self.use_elmo = USE_ELMO
        if not self.use_elmo:
            self.elmo_dim = 0
            
        # define LSTM network to be used on a list level
 
        self.a_lstm = nn.LSTM(embedding_dim + self.elmo_dim, self.SENT_LSTM_DIM, num_layers=self.num_layers, batch_first=True,
                            bidirectional=self.bidirectional, dropout=0.2)        
        self.a_self_attention = self.cent_lstm_att_fn(self.sent_lstm_directions*self.SENT_LSTM_DIM)
        # self.a_layer_norm = BertLayerNorm(hidden_dim*self.sent_lstm_directions)


        self.ctx_bidirectional = True
        self.ctx_lstm_dim = hidden_dim_e
        self.ctx_lstm_directions = 2 if self.ctx_bidirectional else 1

        if not self.add_linear:
            self.deepmoji_out = self.deepmoji_dim

        self.context_lstm = nn.LSTM(self.SENT_LSTM_DIM * self.sent_lstm_directions + self.deepmoji_out, self.ctx_lstm_dim,
                                    num_layers=2, batch_first=True, dropout=0.2, bidirectional=self.ctx_bidirectional)
        self.ctx_self_attention = self.ctx_lstm__att_fn(self.ctx_lstm_directions * self.ctx_lstm_dim)
        # self.ctx_layer_norm = BertLayerNorm(self.ctx_lstm_dim*self.ctx_lstm_directions)
        # self.ctx_pooler = BertPooler(self.ctx_lstm_dim*self.ctx_lstm_directions)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # create a linear model that goes from context lstm dim to prediction
        self.context_to_emo = nn.Linear(self.ctx_lstm_dim, 2)
        self.drop_out = nn.Dropout(0.2)
        self.out2label = nn.Linear(self.ctx_lstm_dim * self.ctx_lstm_directions, 1)
        
        self.sent_pad_len = SENT_PAD_LEN
        self.id2word = id2word
        



    def init_hidden(self, x):
        batch_size = x.size(0)
        if self.bidirectional:
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
        else:
            h0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
        return (h0, c0)

    @staticmethod
    def sort_batch(batch, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        rever_sort = np.zeros(len(seq_lengths))
        for i, l in enumerate(perm_idx):
            rever_sort[l] = i
        return seq_tensor, seq_lengths, rever_sort.astype(int), perm_idx
    
    
    def glove_tokenizer(self, ids, __id2word):
        """
        This function is only called in elmo. Elmo encode is then called
        _id2word: returned by build vocab
        This simply uses id returned from vocab building function 
        """
        return [__id2word[int(x)] for x in ids if x != 0]

    def elmo_encode(self, data, __id2word):
        """
        get the id2word from vocab, then convert to id
        from allennlp.modules.elmo import Elmo, batch_to_ids
        batch_to_id fills to the max sentence length, which could be less than desired
        So further fill it to get to the max sent length
        """
        data_text = [self.glove_tokenizer(x, __id2word) for x in data]
        
        with torch.no_grad():
            elmo = Elmo(options_file, weight_file, 2, dropout=0).cuda()
            elmo.eval()
            character_ids = batch_to_ids(data_text).cuda()
            
            row_num =  character_ids.shape[0]
            elmo_dim = self.elmo_dim
            
            if torch.sum(character_ids) != 0:
                elmo_emb = elmo(character_ids)['elmo_representations']     
                elmo_emb = (elmo_emb[0] + elmo_emb[1]) / 2  # avg of two layers
            else:
                elmo_emb = torch.zeros([row_num, self.sent_pad_len, elmo_dim], dtype=torch.float)
                  
        
        sent_len = elmo_emb.shape[1]        
        
        if sent_len < self.sent_pad_len:
            fill_sent_len = self.sent_pad_len - sent_len
            # create a bunch of 0's to fill it up
            filler = torch.zeros([row_num, fill_sent_len, elmo_dim], dtype=torch.float)
            elmo_emb = torch.cat((elmo_emb, filler.cuda()), dim=1)
        return elmo_emb.cuda()


    def sent_lstm_forward(self, x, x_len,  lstm, hidden=None, attention_layer=None):
        x, x_len_sorted, reverse_idx, perm_idx = self.sort_batch(x, x_len.view(-1))
        
        max_len = int(x_len_sorted[0])
        """
        this only contains the sentence level LSTM
        a_out, a_hidden = self.lstm_forward(a, a_len, elmo_a, self.a_lstm,
                                    attention_layer=self.a_self_attention)
        """

        emb_x = self.embeddings(x)
        emb_x = self.drop_out(emb_x)
#         emb_x = emb_x[:, :max_len, :]

        if self.use_elmo:
            #concatnate elmo and Glove
            elmo_x = self.elmo_encode(x, self.id2word)    
            elmo_x = elmo_x[perm_idx]
            elmo_x = self.drop_out(elmo_x)
            emb_x = torch.cat((emb_x, elmo_x), dim=2)      
                        
        packed_input = nn.utils.rnn.pack_padded_sequence(emb_x, x_len_sorted.cpu().numpy(), batch_first=True)
        if hidden is None:
            hidden = self.init_hidden(x)
            
        packed_output, hidden = lstm(packed_input, hidden)
        output, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # attention_layer = None  # testing
        if attention_layer is None:
            seq_len = torch.LongTensor(unpacked_len).view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
            seq_len = Variable(seq_len - 1).cuda()
            output = torch.gather(output, 1, seq_len).squeeze(1)
        else:
            if isinstance(attention_layer, AttentionOneParaPerChan):
                output, alpha = attention_layer(output, unpacked_len)
            else:
                unpacked_len = [int(x.data) for x in unpacked_len]
                # print(unpacked_len)
                max_len = max(unpacked_len)
                mask = [[1] * l + [0] * (max_len - l) for l in unpacked_len]
                mask = torch.FloatTensor(np.asarray(mask)).cuda()
                attention_mask = torch.ones_like(mask)
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # extended_attention_mask = extended_attention_mask.to(
                #       type=next(self.parameters()).dtype)  # fp16 compatibility
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                output, alpha = attention_layer(output, extended_attention_mask)
                # out, att = self.attention_layer(lstm_out[:, -1:].squeeze(1), lstm_out)
                output = output[:, 0, :]

        return output[reverse_idx], (hidden[0][:, reverse_idx, :], hidden[1][:, reverse_idx, :])
    
    
    def forward(self, list_of_a, list_of_a_len, list_of_a_emoji,  elmo_a=None):        
        """
        This forward the model on the email level, not sentence level
        (a, a_len, emoji_a, e_c) in enumerate(dev_data_loader)
        """
   
                
        # it represents the number of sentences we have padded/cut off for each email
        # we have to append results for each sentence together for the email-wide attention
        NUM_OF_ROWS = list_of_a.shape[0]
        EMAI_PAD_LEN = list_of_a.shape[1]        
        
        # going to concat all the output into one to be put into contex lstm
        context_in = []
        
        for i in range(EMAI_PAD_LEN):
            """
            list_of_a: list of list of words
            list_of_a_len: list of list of lengths 
            a: correspond to an email
            a_len: correspond to sentence lengths in an email
            """
            a = list_of_a[:,i,:]
            a_len = list_of_a_len[:,i]
            a_emoji = list_of_a_emoji[:,i,:]
            
            row_num = len(a)
        
            # elmo_a: whether to use elmo in this model
            # this lstm does not have emoji

            a_out, a_hidden = self.sent_lstm_forward(a, a_len, self.a_lstm,
                                                attention_layer=self.a_self_attention)

            if torch.sum(a_emoji) != 0:
                # a_out = a_out[:, 0, :]
                if self.add_linear:
                    a_emoji = self.deepmoji_model(a_emoji)
                    a_emoji = self.deepmoji2linear(a_emoji)
                    a_emoji = F.relu(a_emoji)
                    a_emoji = self.drop_out(a_emoji)
                else:
                    a_emoji = self.deepmoji_model(a_emoji)
                    a_emoji = F.relu(a_emoji)
            else:
                a_emoji = torch.zeros([row_num, self.deepmoji_dim], dtype=torch.float).cuda()

                
            
            # a_out = torch.cat((F.relu(a_out), a_emoji), dim=1)
            a_out = torch.cat((a_out, a_emoji), dim=1)
            
            context_in.append(a_out.unsqueeze(1))
            
        # Context LSTM
        context_in = torch.cat(context_in, dim=1)
        
        ctx_out, _ = self.context_lstm(context_in)
        ## use contex LSTM to get the h output to be fed into Attention layer
        if isinstance(self.ctx_self_attention, AttentionOneParaPerChan):
            ctx_out, _ = self.ctx_self_attention(ctx_out, torch.LongTensor([EMAI_PAD_LEN for _ in range(NUM_OF_ROWS)]))
        else:
            ctx_out, _ = self.ctx_self_attention(ctx_out)
            ctx_out = ctx_out[:, 0, :]

        # ctx_out = self.ctx_layer_norm(ctx_out)
        # ctx_out = self.ctx_pooler(ctx_out)

        # multi-task learning
        return torch.sigmoid(self.out2label(ctx_out))

    def load_embedding(self, emb):
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
