##################################################################################
# Imports
##################################################################################
# staple imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import glob
import numpy as np
import pandas as pd
import time
import datetime
import argparse
import pdb
import json
import random
import math

# numpy essentials
from numpy import asarray
import numpy as np

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# keras essentials
from keras.preprocessing.sequence import pad_sequences

# sklearn
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix

# pyTorch CRF
from torchcrf import CRF

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AdamW, BertConfig 
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel


# Import data getters
from Models.HelperFunctions import get_packed_padded_output

class SCIBERTPOSAttenCRF(nn.Module):

    def __init__(self, freeze_bert, tokenizer, device, bidirectional):
        super(SCIBERTPOSAttenCRF, self).__init__()
        #Instantiating BERT model object 
        self.scibert_layer = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        
        #Freeze bert layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.scibert_layer.parameters():
                p.requires_grad = False

        self.hidden_dim = 512
        self.hidden_pos = 20
        self.tokenizer = tokenizer
        self.device = device
        self.bidirectional = bidirectional

        # lstm layer for POS tags and the embeddings and attention mechanism
        if bidirectional == True:
            self.lstmpos_layer = nn.LSTM(input_size=44, hidden_size = self.hidden_pos, num_layers = 1, bidirectional=bidirectional, batch_first=True)
            self.self_attention = nn.MultiheadAttention(self.hidden_pos * 2, 1, bias=True) # attention mechanism from PyTorch
            
            self.lstm_layer = nn.LSTM(input_size=808, hidden_size = self.hidden_dim, num_layers = 1, bidirectional=bidirectional, batch_first=True)
            self.hidden2tag = nn.Linear(1024, 2)

        elif bidirectional == False:
            self.lstmpos_layer = nn.LSTM(input_size=44, hidden_size = self.hidden_pos, num_layers = 1, bidirectional=bidirectional, batch_first=True)
            self.self_attention = nn.MultiheadAttention(self.hidden_pos, 1, bias=True)             
            
            self.lstm_layer = nn.LSTM(input_size=788, hidden_size = self.hidden_dim, num_layers = 1, bidirectional=bidirectional, batch_first=True)
            self.hidden2tag = nn.Linear(512, 2)

        # crf
        self.crf_layer = CRF(2, batch_first=True)

    
    def forward(self, input_ids=None, attention_mask=None, labels=None, input_pos=None):

        # one hot encode POS tags
        input_pos = input_pos
        packed_pos_input, pos_perm_idx, pos_seq_lengths = get_packed_padded_output(input_pos.float(), input_ids, attention_mask, self.tokenizer)
        packed_pos_output, (ht_pos, ct_pos) = self.lstmpos_layer(packed_pos_input)

        # Unpack and reorder the output
        pos_output, pos_input_sizes = pad_packed_sequence(packed_pos_output, batch_first=True)
        _, unperm_idx = pos_perm_idx.sort(0)
        lstm_pos_output = pos_output[unperm_idx]
        seq_lengths_ordered = pos_seq_lengths[unperm_idx]
        target = torch.zeros(lstm_pos_output.shape[0], input_pos.shape[1], lstm_pos_output.shape[2])
        target = target.cuda()

        target[:, :lstm_pos_output.shape[1], :] = lstm_pos_output # Expand dimensions of the LSTM transformed pos embeddings

        # Attention over the POS embeddings to up-weigh important features
        attention_applied, attention_weights = self.self_attention( target, target, target, key_padding_mask=None, need_weights=True, attn_mask = None )

        # SCIBERT
        outputs = self.scibert_layer(
            input_ids,
            attention_mask = attention_mask
        )

        # output 0 = batch size 6, tokens 512, each token dimension 768 [CLS] token
        # output 1 = batch size 6, each token dimension 768
        sequence_output = outputs[0]

        #  concatenate scibert and pos tag embeddings
        concatenatedVectors = torch.cat( (sequence_output, attention_applied), 2) # concatenate at dim 2 for embeddings and tags
        concatenatedVectors = concatenatedVectors.cuda()

        # lstm with masks (same as attention masks)
        packed_input, perm_idx, seq_lengths = get_packed_padded_output(concatenatedVectors, input_ids, attention_mask, self.tokenizer)
        packed_output, (ht, ct) = self.lstm_layer(packed_input)

        # Unpack and reorder the output
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = output[unperm_idx] # lstm_output.shape = shorter than the padded torch.Size([6, 388, 512])
        seq_lengths_ordered = seq_lengths[unperm_idx]
        
        # shorten the labels as per the batchsize
        labels = labels[:, :lstm_output.shape[1]]

        # mask the unimportant tokens before log_reg (NOTE: CLS token (position 0) is not masked!!!)
        mask = (
            (input_ids[:, :lstm_output.shape[1]] != self.tokenizer.pad_token_id)
            & (input_ids[:, :lstm_output.shape[1]] != self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
            & (labels != 100)
        )

        # on the first time steps XXX CLS token is active at position 0
        for eachIndex in range( mask.shape[0] ):
            mask[eachIndex, 0] = True

        for eachIndex in range( labels.shape[0] ):
            labels[eachIndex, 0] = 0

        mask_expanded = mask.unsqueeze(-1).expand(lstm_output.size())
        lstm_output *= mask_expanded.float()
        labels *= mask.long()

        # log reg
        probablities = F.relu ( self.hidden2tag( lstm_output ) )
 
        # CRF emissions (without mask)
        loss = self.crf_layer(probablities, labels, reduction='token_mean')

        emissions_ = self.crf_layer.decode( probablities , mask = None)
        emissions = [item for sublist in emissions_ for item in sublist] # flatten the nest list of emissions

        target_emissions = torch.zeros(lstm_output.shape[0], lstm_output.shape[1])
        target_emissions = target_emissions.cuda()
        for eachIndex in range( target_emissions.shape[0] ):
            target_emissions[ eachIndex, :torch.tensor( emissions_[eachIndex] ).shape[0] ] = torch.tensor( emissions_[eachIndex] )
        
        return loss, target_emissions, labels, mask