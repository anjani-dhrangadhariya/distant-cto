'''
Model with SCIBERT as embedding layer and a final CRF decoder
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
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel


# Import data getters
from Models.HelperFunctions import get_packed_padded_output

class SCIBERTCRF(nn.Module):

    def __init__(self, freeze_bert, tokenizer, device, bidirectional):
        super(SCIBERTCRF, self).__init__()
        #Instantiating SciBERT model object 
        self.scibert_layer = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        
        #Freeze SciBERT layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.scibert_layer.parameters():
                p.requires_grad = False

        self.tokenizer = tokenizer

        # log reg
        self.hidden2tag = nn.Linear(768, 2)

        # crf
        self.crf_layer = CRF(2, batch_first=True)

    
    def forward(self, input_ids=None, attention_mask=None, labels=None, input_pos=None):

        # SCIBERT
        outputs = self.scibert_layer(
            input_ids,
            attention_mask = attention_mask
        )

        # output 0 = batch size 6, tokens 512, each token dimension 768 [CLS] token
        # output 1 = batch size 6, each token dimension 768
        # output 2 = layers 13, batch 6 (hidden states), tokens 512, each token dimension 768
        sequence_output = outputs[0] 
        
        # mask the unimportant tokens before log_reg
        mask = (
            (input_ids != self.tokenizer.pad_token_id)
            & (input_ids != self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
            & (labels != 100)
        )
        mask_expanded = mask.unsqueeze(-1).expand(sequence_output.size())
        sequence_output *= mask_expanded.float()
        labels *= mask.long()

        # log reg
        probablities = F.relu ( self.hidden2tag( sequence_output ) )

        # CRF emissions
        loss = self.crf_layer(probablities, labels, reduction='token_mean', mask = None)

        emissions_ = self.crf_layer.decode( probablities , mask = None)
        emissions = [item for sublist in emissions_ for item in sublist] # flatten the nest list of emissions

        target_emissions = torch.zeros(probablities.shape[0], probablities.shape[1])
        target_emissions = target_emissions.cuda()
        for eachIndex in range( target_emissions.shape[0] ):
            target_emissions[ eachIndex, :torch.tensor( emissions_[eachIndex] ).shape[0] ] = torch.tensor( emissions_[eachIndex] )
        
        return loss, target_emissions, labels, mask