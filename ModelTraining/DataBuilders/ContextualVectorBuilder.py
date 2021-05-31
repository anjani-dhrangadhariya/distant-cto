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

# numpy essentials
from numpy import asarray
import numpy as np

# keras essentials
from keras.preprocessing.sequence import pad_sequences

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaConfig, RobertaModel
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup

# visualization
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

##################################################################################
# Set all the seed values
##################################################################################
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def createAttnMask(input_ids):
    # Add attention masks
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return attention_masks

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

# the function truncatedlist of lists to a max length and add special tokens
def truncateSentence(sentence, trim_len):
    """
    Truncates the sequence length to (MAX_LEN - 2). 
    Negating 2 for the special tokens
    """
    trimmedSentence = []
    if  len(sentence) > trim_len:
        trimmedSentence = sentence[:trim_len]
    else:
        trimmedSentence = sentence

    assert len(trimmedSentence) <= trim_len
    return trimmedSentence

# add the special tokens in the end of sequences
def addSpecialtokens(eachText, start_token, end_token):
    insert_at_start = 0
    eachText[insert_at_start:insert_at_start] = [start_token]

    insert_at_end = len(eachText)
    eachText[insert_at_end:insert_at_end] = [end_token]

    assert eachText[0] == start_token
    assert eachText[-1] == end_token

    return eachText

def tokenize_and_preserve_labels(sentence, text_labels, pos, tokenizer, max_length, pretrained_model):

    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """
    dummy_label = 100 # Could be any kind of labels that you can mask
    tokenized_sentence = []
    labels = []
    poss = []
    printIt = []

    for word, label, pos_i in zip(sentence, text_labels, pos):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.encode(word, add_special_tokens = False)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        if n_subwords == 1:
            labels.extend([label] * n_subwords)
            poss.extend( [pos_i] * n_subwords ) 
        else:
            labels.extend([label])
            labels.extend( [dummy_label] * (n_subwords-1) )
            poss.extend( [pos_i] * n_subwords ) 

    assert len(tokenized_sentence) == len(labels) == len(poss)


    # Truncate the sequences (sentence and label) to (max_length - 2)
    if max_length >= 510:
        truncated_sentence = truncateSentence(tokenized_sentence, (max_length - 2))
        truncated_labels = truncateSentence(labels, (max_length - 2))
        truncated_pos = truncateSentence(poss, (max_length - 2))
        assert len(truncated_sentence) == len(truncated_labels) == len(truncated_pos)
    else:
        truncated_sentence = tokenized_sentence
        truncated_labels = labels
        truncated_pos = poss
        assert len(truncated_sentence) == len(truncated_labels) == len(truncated_pos)         

    # Add special tokens CLS and SEP for the BERT tokenizer (identical for SCIBERT)
    if 'bert' in pretrained_model.lower():
        speTok_sentence = addSpecialtokens(truncated_sentence, tokenizer.cls_token_id, tokenizer.sep_token_id)
    elif 'gpt2' in pretrained_model.lower():
        speTok_sentence = addSpecialtokens(truncated_sentence, tokenizer.bos_token_id, tokenizer.eos_token_id)

    speTok_labels = addSpecialtokens(truncated_labels, 0, 0)
    speTok_pos = addSpecialtokens(truncated_pos, 0, 0)

    # PAD the sequences to max length
    if 'bert' in pretrained_model.lower():
        input_ids = pad_sequences([ speTok_sentence ] , maxlen=max_length, value=tokenizer.pad_token_id, padding="post")
    elif 'gpt2' in pretrained_model.lower():
        input_ids = pad_sequences([ speTok_sentence ] , maxlen=max_length, value=tokenizer.unk_token_id, padding="post")

    input_labels = pad_sequences([ speTok_labels ] , maxlen=max_length, value=0, padding="post")
    input_pos = pad_sequences([ speTok_pos ] , maxlen=max_length, value=0, padding="post")

    # Get the attention masks
    attention_masks = createAttnMask( input_ids )
    attention_masks = np.asarray(attention_masks, dtype=np.uint8)

    assert len(input_ids.squeeze()) == max_length
    assert len(input_labels.squeeze()) == max_length
    assert len(attention_masks.squeeze()) == max_length
    assert len(input_pos.squeeze()) == max_length

    return input_ids.squeeze(), input_labels.squeeze(), attention_masks.squeeze(), input_pos.squeeze()


def get_contextual_vectors(annotations_df, vector_type, pos_encoder, MAX_LEN):

    # Choose tokenizer here
    tokenizer = choose_tokenizer_type(vector_type)

    # XXX Training set: tokenize, preserve labels, truncate, add special tokens and pad to the MAX_LEN_new
    tokenized = []
    for tokens, labels, pos in zip(list(annotations_df['tokens']), list(annotations_df['labels']), list(annotations_df['pos'])) :
        temp = tokenize_and_preserve_labels(tokens, labels, pos_encoder.transform(pos), tokenizer, MAX_LEN, vector_type)
        tokenized.append( temp ) # coarse labels for the training set


    tokens, labels, masks, poss = list(zip(*tokenized))
    
    return tokens, labels, masks, poss # Returns input IDs and labels together