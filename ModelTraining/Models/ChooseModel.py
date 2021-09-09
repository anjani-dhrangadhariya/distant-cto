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
import sys, json, os
import logging
import datetime as dt
import time

# sklearn
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaConfig, RobertaModel
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Visualization
from tqdm import tqdm

from Models.BERT_CRF import BERTCRF
from Models.BERT_Linear import BERTLinear
from Models.SCIBERT_CRF import SCIBERTCRF
from Models.SCIBERTPOS_CRF import SCIBERTPOSCRF
from Models.SCIBERT_Linear import SCIBERTLinear
from Models.SCIBERTPOS_Linear import SCIBERTPOSLinear
from Models.SCIBERTPOSATTEN_CRF import SCIBERTPOSAttenCRF
from Models.SCIBERTPOSATTEN_Linear import SCIBERTPOSAttenLinear
from Models.SCIBERTPOSATTEN_activation import SCIBERTPOSAttenActLin
from Models.BERTBiLSTM_CRF import BERTBiLSTMCRF
from Models.semantic_crf import SemanticCRF

def choose_tokenizer_type(pretrained_model):
    ##################################################################################
    # Load the BERT tokenizer XXX Set tokenizer
    ##################################################################################
    print('Loading BERT tokenizer...')
    if 'bert' in pretrained_model and 'bio' not in pretrained_model and 'sci' not in pretrained_model:
        tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    elif 'gpt2' in pretrained_model:
        tokenizer_ = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True, unk_token="<|endoftext|>")

    elif 'biobert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    elif 'scibert' in pretrained_model:
        tokenizer_ = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    return tokenizer_

def choose_model(vector_type, pretrained_model, args):

    tokenizer = choose_tokenizer_type( vector_type )   

    if pretrained_model == 'bertcrf':
        model = BERTCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    if pretrained_model == 'bertbilstmcrf':
        model = BERTBiLSTMCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    elif pretrained_model == 'bertlinear':
        model = BERTLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    elif pretrained_model == 'scibertcrf':
        model = SCIBERTCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    elif pretrained_model == 'scibertposcrf':
        model = SCIBERTPOSCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    elif pretrained_model == 'scibertposlinear':
        model = SCIBERTPOSLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    elif pretrained_model == 'scibertlinear':
        model = SCIBERTLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    elif pretrained_model == 'scibertposattencrf':
        model = SCIBERTPOSAttenCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    elif pretrained_model == 'scibertposattenlinear':
        model = SCIBERTPOSAttenLinear(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    elif pretrained_model == 'scibertposattenact':
        model = SCIBERTPOSAttenActLin(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    # elif pretrained_model == 'semanticcrf':
    #     model = SemanticCRF(args.freeze_bert, tokenizer, args.gpu, args.bidrec)
    else:
        print('Please enter correct model name...')


    return model