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

class SemanticCRF(nn.Module):

    def __init__(self):
        super(SemanticCRF, self).__init__()

        

        # log reg
        self.hidden2tag = nn.Linear(300, 2)

        # crf
        self.crf_layer = CRF(2, batch_first=True)

        # binary cross entropy loss
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, input_ids=None, attention_mask=None, labels=None):

        # log reg
        probablities_ = F.softmax ( self.hidden2tag( input_ids.float() ) )
       
        labels = labels[attention_mask]
        probablities___ = probablities_[attention_mask,:]

        loss = self.criterion( probablities___.view(-1, 2), labels.view(-1) )
        return loss, probablities___, labels

        # CRF emissions
        # loss = self.crf_layer(probablities, labels, reduction='token_mean', mask=attention_mask)
        # emissions = self.crf_layer.decode( probablities )
        # emissions_flattened = [item for sublist in emissions for item in sublist] # flatten the nest list of emissions
        # return loss, torch.Tensor(emissions_flattened), labels