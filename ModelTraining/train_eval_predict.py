##################################################################################
# Imports
##################################################################################
# staple imports
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import datetime
import datetime as dt
# Memory leak
import gc
import glob
import json
import logging
import os
import pdb
import random
import shutil
# statistics
import statistics
import sys
import time
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

# sklearn
import sklearn

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, plot_confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# Torch modules
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# Visualization
from tqdm import tqdm

# Transformers 
from transformers import (AdamW, AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer, BertConfig, BertModel, BertTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer, RobertaConfig,
                          RobertaModel, get_linear_schedule_with_warmup)

warnings.filterwarnings('ignore')

import mlflow

from Utilities.mlflow_logging import *

def printMetrics(cr, args):

    if args.num_labels == 5:   
        return cr['macro avg']['f1-score'], cr['1']['f1-score'], cr['2']['f1-score'], cr['3']['f1-score'], cr['4']['f1-score']
    elif args.num_labels == 4:   
        return cr['macro avg']['f1-score'], cr['1']['f1-score'], cr['2']['f1-score'], cr['3']['f1-score']
    elif args.num_labels == 2:
        return cr['macro avg']['f1-score'], cr['1']['f1-score']


def flattenIt(x):

    return np.asarray(x.cpu(), dtype=np.float32).flatten()


def evaluate(defModel, optimizer, scheduler, development_dataloader, exp_args, epoch_number = None):
    mean_acc = 0
    mean_loss = 0
    count = 0
    total_val_loss_coarse = 0

    with torch.no_grad() :
        # collect all the evaluation predictions and ground truth here
        all_GT = []
        all_masks = []
        all_predictions = []
        all_tokens = []

        class_rep_temp = []

        for e_input_ids_, e_labels, e_input_mask, e_input_pos in development_dataloader:

            e_input_ids_ = e_input_ids_.to(f'cuda:{defModel.device_ids[0]}')

            with torch.cuda.device_of(e_input_ids_.data): # why am I cloning this variable?
                e_input_ids = e_input_ids_.clone()

            # load the variables on the device
            e_input_mask = e_input_mask.to(f'cuda:{defModel.device_ids[0]}')
            e_labels = e_labels.to(f'cuda:{defModel.device_ids[0]}')
            e_input_pos = e_input_pos.to(f'cuda:{defModel.device_ids[0]}')

            e_loss, e_output, e_labels, e_mask = defModel(e_input_ids, attention_mask=e_input_mask, labels=e_labels, input_pos=e_input_pos) 

            # shorten the input_ids to match the e_output shape (This is to retrieve the natural langauge words from the input IDs)
            e_input_ids = e_input_ids[:, :e_output.shape[1]]

            if len(list( e_output.shape )) > 1:
                if 'crf' not in exp_args.model:
                    max_probs = torch.max(e_output, dim=2) # get the highest of two probablities
                    e_logits = max_probs.indices
                else:
                    e_logits = e_output 
            else: 
                e_logits = e_output

            mean_loss += abs( torch.mean(e_loss) ) 

            for i in range(0, e_labels.shape[0]):

                masked_preds = torch.masked_select( e_logits[i, ].to(f'cuda:0'), e_mask[i, ] )
                masked_labs = torch.masked_select( e_labels[i, ].to(f'cuda:0'), e_mask[i, ] )

                temp_cr = classification_report(y_pred= masked_preds.cpu(), y_true=masked_labs.cpu(), labels=list(range(exp_args.num_labels)), output_dict=True) 
                class_rep_temp.append(temp_cr['macro avg']['f1-score'])

                all_masks.extend( e_mask[i, ] )
                all_GT.extend( e_labels[i, ] )
                all_predictions.extend( e_logits[i, ] )
                all_tokens.extend( e_input_ids[i, ] )          

        avg_val_loss = mean_loss / len(development_dataloader)

        # stack the list of tensors into a tensor
        all_masks_tens = torch.stack(( all_masks ))
        all_GT_tens =  torch.stack(( all_GT ))
        all_preds_tens = torch.stack(( all_predictions ))
        all_token_tens = torch.stack(( all_tokens ))

        # mask the prediction tensor
        selected_preds_coarse = torch.masked_select( all_preds_tens.to(f'cuda:0'), all_masks_tens )
        # mask the label tensor
        selected_labs_coarse = torch.masked_select( all_GT_tens.to(f'cuda:0'), all_masks_tens )
        # mask the natural language token tensor but with attention mask carefully
        selected_tokens_coarse = torch.masked_select( all_token_tens.to(f'cuda:0'), all_masks_tens )   

        # flatten the masked tensors
        all_pred_flat = flattenIt( selected_preds_coarse )
        # all_pred_flat = np.asarray(selected_preds_coarse.cpu(), dtype=np.float32).flatten()
        all_GT_flat = flattenIt( selected_labs_coarse )
        # all_GT_flat = np.asarray(selected_labs_coarse.cpu(), dtype=np.float32).flatten()
        all_tokens_flat = flattenIt( selected_tokens_coarse )
        # all_tokens_flat = np.asarray(selected_tokens_coarse.cpu(), dtype=np.int64).flatten()

        # Final classification report and confusion matrix for each epoch
        val_cr = classification_report(y_pred= all_pred_flat, y_true=all_GT_flat, labels=list(range(exp_args.num_labels)), output_dict=True)

        # confusion_matrix and plot
        labels = [0, 1]
        cm = confusion_matrix(all_GT_flat, all_pred_flat, labels, normalize=None)

    return val_cr, all_pred_flat, all_GT_flat, cm, all_tokens_flat, class_rep_temp        

                                
# Train
def train(defModel, optimizer, scheduler, train_dataloader, development_dataloader, exp_args, run, eachSeed):

    saved_models = []

    with torch.enable_grad():
        best_f1 = 0.0

        for epoch_i in range(0, exp_args.max_eps):
            # Accumulate loss over an epoch
            total_train_loss = 0

            # accumulate predictions and labels over the epoch
            train_epoch_logits_coarse_i = np.empty(1, dtype=np.int64)
            train_epochs_labels_coarse_i = np.empty(1, dtype=np.int64)

            # Training for all the batches in this epoch
            for step, batch in enumerate(train_dataloader):

                # Clear the gradients
                optimizer.zero_grad()

                b_input_ids = batch[0].to(f'cuda:{defModel.device_ids[0]}')
                b_labels = batch[1].to(f'cuda:{defModel.device_ids[0]}')
                b_masks = batch[2].to(f'cuda:{defModel.device_ids[0]}')
                b_pos = batch[3].to(f'cuda:{defModel.device_ids[0]}')

                b_loss, b_output, b_labels, b_mask = defModel(input_ids = b_input_ids, attention_mask=b_masks, labels=b_labels, input_pos=b_pos)

                total_train_loss += abs( torch.mean(b_loss) ) 

                abs( torch.mean(b_loss) ).backward()

                # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(defModel.parameters(), 1.0)

                #Optimization step
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                for i in range(0, b_labels.shape[0]): # masked select excluding the post padding 

                    selected_preds_coarse = torch.masked_select( b_output[i, ].to(f'cuda:{defModel.device_ids[0]}'), b_mask[i, ])
                    selected_labs_coarse = torch.masked_select(b_labels[i, ].to(f'cuda:{defModel.device_ids[0]}'), b_mask[i, ])

                    train_epoch_logits_coarse_i = np.append(train_epoch_logits_coarse_i, selected_preds_coarse.to("cpu").numpy(), 0)
                    train_epochs_labels_coarse_i = np.append(train_epochs_labels_coarse_i, selected_labs_coarse.to("cpu").numpy(), 0)

                if step % exp_args.print_every == 0:
                    cr = sklearn.metrics.classification_report(y_pred= train_epoch_logits_coarse_i, y_true= train_epochs_labels_coarse_i, labels= list(range(exp_args.num_labels)), output_dict=True)
                    if exp_args.num_labels == 5:  
                        f1, f1_1 , f1_2, f1_3, f1_4 = printMetrics(cr, exp_args)
                        print('Training: Epoch {} with macro average F1: {}, F1 score (P): {}, F1 score (IC): {}, F1 score (O): {}, F1 score (S): {}'.format(epoch_i, f1, f1_1 , f1_2, f1_3, f1_4))
                    elif exp_args.num_labels == 4:  
                        f1, f1_1 , f1_2, f1_3 = printMetrics(cr, exp_args)
                        print('Training: Epoch {} with macro average F1: {}, F1 score (P): {}, F1 score (IC): {}, F1 score (O): {}'.format(epoch_i, f1, f1_1 , f1_2, f1_3))
                    elif exp_args.num_labels == 2:
                        f1, f1_1 = printMetrics(cr, exp_args)
                        print('Training: Epoch {} with macro average F1: {}, F1 score (entity): {}'.format(epoch_i, f1, f1_1))


            ## Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            train_cr = classification_report(y_pred= train_epoch_logits_coarse_i, y_true=train_epochs_labels_coarse_i, labels= list(range(exp_args.num_labels)), output_dict=True)             

            # Delete the collected logits and labels
            #del train_epoch_logits_coarse_i, train_epochs_labels_coarse_i
            #gc.collect()

            val_cr, all_pred_flat_coarse, all_GT_flat_coarse, cm, all_tokens_flat, class_rep_temp  = evaluate(defModel, optimizer, scheduler, development_dataloader, exp_args, epoch_i)
            if exp_args.num_labels == 5:  
                val_f1, val_f1_1 , val_f1_2, val_f1_3, val_f1_4 = printMetrics(val_cr, exp_args)
                logMetrics("val f1", val_f1, epoch_i)
                logMetrics("val f1 P", val_f1_1, epoch_i); logMetrics("val f1 IC", val_f1_2, epoch_i); logMetrics("val f1 O", val_f1_3, epoch_i); logMetrics("val f1 S", val_f1_4, epoch_i)
                print('Validation: Epoch {} with macro average F1: {}, F1 score (P): {}, F1 score (IC): {}, F1 score (O): {}, F1 score (S): {}'.format(epoch_i, val_f1, val_f1_2 , val_f1_2, val_f1_3, val_f1_4))

            elif exp_args.num_labels == 4:  
                val_f1, val_f1_1 , val_f1_2, val_f1_3 = printMetrics(val_cr, exp_args)
                logMetrics("val f1", val_f1, epoch_i)
                logMetrics("val f1 P", val_f1_1, epoch_i); logMetrics("val f1 IC", val_f1_2, epoch_i); logMetrics("val f1 O", val_f1_3, epoch_i)
                print('Validation: Epoch {} with macro average F1: {}, F1 score (P): {}, F1 score (IC): {}, F1 score (O): {}'.format(epoch_i, val_f1, val_f1_2 , val_f1_2, val_f1_3))
            elif exp_args.num_labels == 2:
                val_f1, val_f1_1 = printMetrics(val_cr, exp_args)
                logMetrics("val f1", val_f1, epoch_i)
                string2print = "val f1" + str(exp_args.entity)
                logMetrics(string2print, val_f1_1, epoch_i)
                print('Validation: Epoch {} with macro average F1: {}, F1 score: {}'.format(epoch_i, val_f1, val_f1_1))

            # Process of saving the model
            if val_f1 > best_f1:

                base_path = "/mnt/nas2/results/Models/systematicReview/Distant_CTO/models/" + str(exp_args.entity) + "/" + str(exp_args.model)
                if not os.path.exists( base_path ):
                    oldmask = os.umask(000)
                    os.makedirs(base_path)
                    os.umask(oldmask)

                print("Best validation F1 improved from {} to {} ...".format( best_f1, val_f1 ))
                model_name_here = base_path + '/' +str(eachSeed) + '_' + str(exp_args.annot) + '2_' + str(exp_args.embed) + '_epoch_' + str(epoch_i) + '.pth'
                print('Saving the best model for epoch {} with mean F1 score of {} '.format(epoch_i, val_f1 )) 
                torch.save(defModel.state_dict(), model_name_here)
                best_f1 = val_f1
                saved_models.append(model_name_here)

    return saved_models