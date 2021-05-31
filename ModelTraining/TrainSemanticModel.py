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


from ReadCandidateData import *
from Models.semantic_crf import SemanticCRF

##################################################################################
# set up the GPU
##################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print('Number of GPUs identified: ', n_gpu)
print(torch.cuda.get_device_name(0))
print('You are using ', torch.cuda.get_device_name(0), ' : ', device , ' device')

##################################################################################
# set all the seed values
##################################################################################
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Evaluate
def evaluate(defModel, optimizer, scheduler, development_dataloader):
    mean_acc = 0
    mean_loss = 0
    count = 0
    total_val_loss_coarse = 0

    with torch.no_grad() :

        # collect all the evaluation predictions and ground truth here
        all_predictions = []
        all_GT = []

        for e_input_ids, e_labels, e_input_mask in development_dataloader:
            e_loss, e_output, e_labels  = defModel(e_input_ids, attention_mask=e_input_mask, labels=e_labels)

            mean_loss += e_loss.item()

            for i in range(0, e_labels.shape[0]):
                # selected_preds = torch.masked_select(e_output[i, ], e_input_mask[i, ])
                # selected_labs = torch.masked_select(e_labels[i, ], e_input_mask[i, ])
                all_predictions.extend( torch.argmax(e_output, dim=1) )
                all_GT.extend( e_labels )


        # Final classification report and confusion matrix for each epoch
        all_pred_flat = np.asarray(all_predictions).flatten()
        all_GT_flat = np.asarray(all_GT).flatten()
        val_cr = classification_report(y_pred=all_pred_flat, y_true=all_GT_flat, labels=list(range(2)), output_dict=True)     

    return val_cr, all_pred_flat, all_GT_flat


# Train
def train(defModel, optimizer, scheduler, train_dataloader, development_dataloader, args):

    with torch.enable_grad():
        best_meanf1 = 0.0
        for epoch_i in range(0, args.max_eps):
            # Accumulate loss over an epoch
            total_train_loss = 0

            # (coarse-grained) accumulate predictions and labels over the epoch
            train_epoch_logits_coarse = []
            train_epochs_labels_coarse = []

            # Training for all the batches in this epoch
            for step, batch in enumerate(train_dataloader):

                # Clear the gradients
                optimizer.zero_grad()

                b_input_ids = batch[0]
                b_labels = batch[1]
                b_masks = batch[2]

                b_loss, b_output, b_labels = defModel(input_ids = b_input_ids, attention_mask=b_masks, labels=b_labels)

                total_train_loss += abs( b_loss.item() )

                abs(b_loss).backward()

                # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(defModel.parameters(), 1.0)

                #Optimization step
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                for i in range(0, b_labels.shape[0]): 

                    # selected_preds_coarse = torch.masked_select(b_output[i, ], b_masks[i, ])
                    # selected_labs_coarse = torch.masked_select(b_labels[i, ], b_masks[i, ])

                    train_epoch_logits_coarse.extend( torch.argmax(b_output, dim=1) )
                    train_epochs_labels_coarse.extend( b_labels )



                if step % args.print_every == 0:

                    cr = classification_report(y_pred=train_epoch_logits_coarse, y_true=train_epochs_labels_coarse, labels= list(range(2)), output_dict=True)
                    meanF1_1 = cr['1']['f1-score']
                    meanF1_0 = cr['0']['f1-score']
                    print('Training: Epoch {} with mean F1 score (1): {}, mean F1 score (0): {}'.format(epoch_i, meanF1_1, meanF1_0))

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            val_cr, all_pred_flat_coarse, all_GT_flat_coarse = evaluate(defModel, optimizer, scheduler, development_dataloader)
            val_meanF1_1 = val_cr['1']['f1-score']
            val_meanF1_0 = val_cr['0']['f1-score']
            # print('Validation F1 for class intervention ', val_meanF1)
            print('\nValidation: Epoch {} with mean F1 score (1): {}, mean F1 score (0): {} and traing loss : {}'.format(epoch_i, val_meanF1_1, val_meanF1_0, avg_train_loss))


##########################################################################
# define parameter space here
##########################################################################
vectorization = 'semantic'

if __name__ == "__main__":

    annotations = FetchTrainingCandidates()
    # FetchTrainingCandidates()


    ##########################################################################
    # Instead of random split use K-Fold function from Sklearn
    ##########################################################################
    kf = KFold(n_splits=10)
    for fold, (train_index, dev_index) in enumerate( kf.split(annotations) ):

        if fold == 0:

            X_train, X_dev = np.array( list(annotations['embeddings']), dtype=np.float32)[train_index], np.array( list(annotations['embeddings']), dtype=np.float32)[dev_index]
            y_train, y_dev = np.array( list(annotations['label_pads']), dtype=np.int64)[train_index], np.array( list(annotations['label_pads']), dtype=np.int64)[dev_index]
            train_masks, dev_masks = np.array( list(annotations['attn_masks']), dtype=np.int64)[train_index], np.array( list(annotations['attn_masks']), dtype=np.int64)[dev_index]

            # Convert all inputs, labels, and attentions into torch tensors, the required datatype: torch.int64
            train_input_ids = torch.tensor(torch.from_numpy(X_train), dtype=torch.int64)                    
            train_input_labels = torch.tensor(torch.from_numpy(y_train), dtype=torch.int64)
            train_attn_masks = torch.tensor(torch.from_numpy(train_masks), dtype=torch.bool)

            dev_input_ids = torch.tensor(torch.from_numpy(X_dev), dtype=torch.int64)                    
            dev_input_labels = torch.tensor(torch.from_numpy(y_dev), dtype=torch.int64)
            dev_attn_masks = torch.tensor(torch.from_numpy(dev_masks), dtype=torch.bool)


            print('Inputs converted to tensors...')

            # Create the DataLoader for our training set.
            train_data = TensorDataset(train_input_ids, train_input_labels, train_attn_masks)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=None, batch_size=20, shuffle=False)

            # Create the DataLoader for our development set.
            dev_data = TensorDataset(dev_input_ids, dev_input_labels, dev_attn_masks)
            dev_sampler = RandomSampler(dev_data)
            development_dataloader = DataLoader(dev_data, sampler=None, batch_size=20, shuffle=False)

            print('\n--------------------------------------------------------------------------------------------')
            print('Data loaders created')
            print('--------------------------------------------------------------------------------------------')

            ##################################################################################
            #Instantiating the BERT model
            ##################################################################################
            print("Building model...")
            st = time.time()
            model = SemanticCRF()
            print("Done in {} seconds".format(time.time() - st))

            ##################################################################################
            # Parse the arguments for training
            ##################################################################################
            parser = argparse.ArgumentParser()
            parser.add_argument('-gpu', type = int, default = device)
            parser.add_argument('-freeze_bert', action='store_false') # store_false = won't freeze BERT
            parser.add_argument('-print_every', type = int, default= 40)
            parser.add_argument('-max_eps', type = int, default= 100)
            parser.add_argument('-lr', type = float, default= 5e-4)
            parser.add_argument('-eps', type = float, default= 1e-8)
            parser.add_argument('-loss', type = str, default = 'general')
            parser.add_argument('-fold', type = str, default=fold)
            args = parser.parse_args()

            ##################################################################################
            # Set up the optimizer and the scheduler
            ##################################################################################
            st = time.time()
            # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
            # optimizer = AdamW(model.parameters(),
            #                 lr = args.lr, # args.learning_rate - default is 5e-5 (for BERT-base)
            #                 eps = args.eps, # args.adam_epsilon  - default is 1e-8.
            #                 )
            optimizer = optim.SGD(model.parameters(), lr=args.lr)

            # Total number of training steps is number of batches * number of epochs.
            total_steps = len(train_dataloader) * args.max_eps
            print('Total steps per epoch: ', total_steps)

            # Create the learning rate scheduler.
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=0,
                                                        num_training_steps = total_steps)

            print("Created the optimizer, scheduler and loss function objects in {} seconds".format(time.time() - st))


            print('##################################################################################')
            print('Begin training...')
            print('##################################################################################')
            train(model, optimizer, scheduler, train_dataloader, development_dataloader, args)
            print("Training and validation done in {} seconds".format(time.time() - st))