'''
Model with BERT as embedding layer followed by a linear decoder
'''
__author__ = "Anjani Dhrangadhariya"
__maintainer__ = "Anjani Dhrangadhariya"
__email__ = "anjani.k.dhrangadhariya@gmail.com"
__status__ = "Prototype/Research"

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

class BERTLinear(nn.Module):

    def __init__(self, freeze_bert, tokenizer, device, bidirectional):
        super(BERTLinear, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)
        
        #Freeze bert layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.tokenizer = tokenizer
        self.device = device
        self.bidirectional = bidirectional

        # lstm layer
        self.lstm_layer = nn.LSTM(input_size=768, hidden_size = 512, num_layers = 1, bidirectional=bidirectional, batch_first=True)

        # log reg
        if bidirectional == True:
            self.hidden2tag = nn.Linear(1024, 2)
        else:
            self.hidden2tag = nn.Linear(512, 2)

        # loss calculation
        self.loss_fct = nn.CrossEntropyLoss()

    
    def forward(self, input_ids=None, attention_mask=None, labels=None, input_pos=None):

        # BERT
        outputs = self.bert_layer(
            input_ids,
            attention_mask = attention_mask
        )

        # output 0 = batch size 6, tokens 512, each token dimension 768 [CLS] token
        # output 1 = batch size 6, each token dimension 768
        # output 2 = layers 13, batch 6 (hidden states), tokens 512, each token dimension 768
        sequence_output = outputs[0]

        # lstm with masks (same as attention masks)
        packed_input, perm_idx, seq_lengths = get_packed_padded_output(sequence_output, input_ids, attention_mask, self.tokenizer)
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

        mask_expanded = mask.unsqueeze(-1).expand(lstm_output.size())
        lstm_output *= mask_expanded.float()
        labels *= mask.long()

        # Linear transofrmation
        probablity = F.relu ( self.hidden2tag( lstm_output ) )
        max_probs = torch.max(probablity, dim=2)         
        logits = max_probs.indices.flatten()

        # Cross-entropy loss calculation
        masked_probs = torch.masked_select( probablity.view( (-1, 2) ) , mask.view((-1)).unsqueeze(-1) )
        masked_labs = torch.masked_select(labels, mask)
        loss = self.loss_fct( masked_probs.view((-1, 2)) , masked_labs  )

        return loss, probablity, labels, mask
