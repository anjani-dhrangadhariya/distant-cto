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
from sklearn.model_selection import train_test_split

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
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from ReadCandidateData import *
from Models.semantic_crf import SemanticCRF
from Models.ChooseModel import choose_model
from Functions.ThresholdMoving import getRoc

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

##################################################################################
# Helper functions
##################################################################################
def norm(x):
    # normalise x to range [-1,1]
    nom = (x - x.min()) * 2.0
    denom = x.max() - x.min()
    return  nom/denom - 1.0

def sigmoid(x, k=0.1):
    # sigmoid function
    # use k to adjust the slope
    s = 1 / (1 + np.exp(-x / k)) 
    return s

# Evaluate
def evaluate(defModel, optimizer, scheduler, development_dataloader, args, exp_args, epoch_number = None):
    mean_acc = 0
    mean_loss = 0
    count = 0
    total_val_loss_coarse = 0

    with torch.no_grad() :

        # collect all the evaluation predictions and ground truth here
        raw_probablities = []
        all_GT = []
        all_masks = []
        all_predictions = []

        for e_input_ids_, e_labels, e_input_mask, e_input_pos in development_dataloader:

            e_input_ids_ = e_input_ids_.cuda()

            with torch.cuda.device_of(e_input_ids_.data):
                e_input_ids = e_input_ids_.clone()

            # check if the model is GPT2-based
            if 'gpt2' in exp_args.embed:
                e_input_ids[e_input_ids_==0] = tokenizer.unk_token_id
                e_input_ids[e_input_ids_==101] = tokenizer.bos_token_id
                e_input_ids[e_input_ids_==102] = tokenizer.eos_token_id

            # coarse grained entity labels
            e_input_mask = e_input_mask.cuda()
            e_labels = e_labels.cuda()
            e_input_pos = e_input_pos.cuda()

            e_loss, e_output, e_labels, e_mask = defModel(e_input_ids, attention_mask=e_input_mask, labels=e_labels, input_pos=e_input_pos)


            mean_loss += e_loss.item()

            for i in range(0, e_labels.shape[0]):
                # select_rawprobas = torch.masked_select( e_output[i, ].cuda(), mask_expanded )
                # selected_labs = torch.masked_select(e_labels[i, ].cuda(), e_mask[i, ])

                max_probs = torch.max(e_output, dim=2)

                raw_probablities.extend(  e_output[i, ] )
                all_predictions.extend( max_probs.values[i, ] )
                all_GT.extend( e_labels[i, ] )
                all_masks.extend( e_mask[i, ] )
            count += 1

        avg_val_loss = mean_loss / len(development_dataloader)     
        # writer.add_scalar('loss-validation', avg_val_loss, epoch_number) # XXX Return the avg_val_loss from this method to the train method and write to the writer from there

        # Mask raw probablities and labels
        raw_probas_arr =  torch.stack(( raw_probablities ))
        mask_arr =  torch.stack(( all_masks ))
        mask_expanded = mask_arr.unsqueeze(-1).expand( raw_probas_arr.size() )
        print( mask_expanded )
        masked_output = raw_probas_arr * mask_expanded.int().float()
        print( masked_output )
        all_GT_arr =  torch.stack(( all_GT ))
        selected_labs = torch.masked_select(all_GT_arr, mask_arr)

        # Normalize the raw probablities
        normalized_probas = norm( masked_output )
        sigrawprobas_output = sigmoid( normalized_probas.cpu() )
        print( sigrawprobas_output )

        all_GT_flat = np.asarray(selected_labs, dtype=np.float32).flatten()
        all_rawprobas_flat = np.asarray(sigrawprobas_output , dtype=np.float32).flatten() 

        # Get the best threshold
        getRoc(all_GT_flat, all_rawprobas_flat)

        # Final classification report and confusion matrix for each epoch
        ### If working on a sentence selector, then only include predictions 
        val_cr = classification_report(y_pred=all_pred_flat, y_true=all_GT_flat, labels=list(range(2)), output_dict=True)     

        # confusion_matrix and plot
        labels = [1,0]
        cm = confusion_matrix(all_GT_flat, all_pred_flat, labels)

    return val_cr, all_pred_flat, all_GT_flat, cm


# Train
def train(defModel, optimizer, scheduler, train_dataloader, development_dataloader, args, exp_args):

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

                b_input_ids = batch[0].cuda()
                b_labels = batch[1].cuda()
                b_masks = batch[2].cuda()
                b_pos = batch[3].cuda()

                b_loss, b_output, b_labels, b_mask = defModel(input_ids = b_input_ids, attention_mask=b_masks, labels=b_labels, input_pos=b_pos)

                total_train_loss += abs( b_loss.item() )

                abs(b_loss).backward()

                # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(defModel.parameters(), 1.0)

                #Optimization step
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                for i in range(0, b_labels.shape[0]): 

                    selected_preds_coarse = torch.masked_select(b_output[i, ].cuda(), b_mask[i, ])
                    selected_labs_coarse = torch.masked_select(b_labels[i, ].cuda(), b_mask[i, ])

                    train_epoch_logits_coarse.extend( selected_preds_coarse.to("cpu").numpy() )
                    train_epochs_labels_coarse.extend( selected_labs_coarse.to("cpu").numpy() )

                if step % args.print_every == 0:

                    cr = classification_report(y_pred=train_epoch_logits_coarse, y_true=train_epochs_labels_coarse, labels= list(range(2)), output_dict=True)
                    # meanF1_2 = cr['2']['f1-score']
                    meanF1_1 = cr['1']['f1-score']
                    meanF1_0 = cr['0']['f1-score']
                    # mean_1 = (meanF1_1 + meanF1_2) / 2
                    print('Training: Epoch {} with mean F1 score (1): {}, mean F1 score (0): {}'.format(epoch_i, meanF1_1 , meanF1_0))

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            writer.add_scalar('loss-train', avg_train_loss, epoch_i)
            train_cr = classification_report(y_pred= train_epoch_logits_coarse, y_true=train_epochs_labels_coarse, labels= list(range(2)), output_dict=True) 
            # meanF1_2 = train_cr['2']['f1-score']
            meanF1_1 = train_cr['1']['f1-score']
            meanF1_0 = train_cr['0']['f1-score']
            mean_1 = (meanF1_1 + meanF1_0) / 2
            writer.add_scalar('f1-train', meanF1_1  , epoch_i)

            # Validation 
            val_cr, all_pred_flat_coarse, all_GT_flat_coarse, cm = evaluate(defModel, optimizer, scheduler, development_dataloader, args, exp_args, epoch_i)
            # val_meanF1_2 = val_cr['2']['f1-score']
            val_meanF1_1 = val_cr['1']['f1-score']
            val_meanF1_0 = val_cr['0']['f1-score']
            val_mean_1 = (val_meanF1_1 + val_meanF1_0) / 2
            # print('Validation F1 for class intervention ', val_meanF1)
            print('Validation: Epoch {} with mean F1 score (1): {}, mean F1 score (0): {} and traing loss : {}'.format(epoch_i, val_meanF1_1, val_meanF1_0, avg_train_loss))
            writer.add_scalar('f1-validation', val_meanF1_1 , epoch_i)

            if val_meanF1_1 > best_meanf1:
                print("Best validation mean F1 improved from {} to {} ...".format( best_meanf1, val_meanF1_1 ))
                model_name_here = '/mnt/nas2/results/Results/systematicReview/Distant_CTO/models/intervention/noEBMTrain_w_shortAnnot_SCIPOSAtten_crf/' + 'bert_bilstm_crf_epoch_' + str(epoch_i) + '_best_model.pth'
                # model_name_here = ''
                print('Saving the best model for epoch {} with mean F1 score of {} '.format(epoch_i, val_meanF1_1 )) 
                torch.save(defModel.state_dict(), model_name_here)
                saved_models.append(model_name_here)                     
                best_meanf1 = val_meanF1_1


##########################################################################
# define parameter space here
##########################################################################
saved_models = []

if __name__ == "__main__":

    # Week annotations, Crowd annotations from EBM-NLP, Gold annotations from medical professionals, Annotations from physiotherapy articles
    # annotations = week, annotations_testdf_ = EBNLP_training, annotations_test1df_ = EBMNLP_test, annotations_test2df_ = hilfiker
    annotations, annotations_testdf_, annotations_test1df_, annotations_test2df_, exp_args = FetchTrainingCandidates()

    # Combine the training (annotations) and development (annotations_testdf_) dataframes, shuffle them and divide into training and validation sets
    #fulldf = annotations.append(annotations_testdf_, ignore_index=True)
    fulldf = annotations
    fulldf = fulldf.sample(frac=1).reset_index(drop=True)
    annotations, annotations_testdf_ = train_test_split(fulldf, test_size=0.2) 
    print('Size of training set: ', len(annotations.index))
    print('Size of development set: ', len(annotations_testdf_.index))

    # #########################################################################
    # Instead of random split use K-Fold function from Sklearn
    # #########################################################################
    # kf = KFold(n_splits=10)
    # for fold, (train_index, dev_index) in enumerate( kf.split(annotations) ):

    #     if fold == 0:

            # X_train, X_dev = np.array( list(annotations['embeddings']), dtype=np.float32)[train_index], np.array( list(annotations['embeddings']), dtype=np.float32)[dev_index]
            # y_train, y_dev = np.array( list(annotations['label_pads']), dtype=np.int64)[train_index], np.array( list(annotations['label_pads']), dtype=np.int64)[dev_index]
            # train_masks, dev_masks = np.array( list(annotations['attn_masks']), dtype=np.int64)[train_index], np.array( list(annotations['attn_masks']), dtype=np.int64)[dev_index]

    # ----------------------------------------------------------------------------------------
    # Load the data into tensors
    # ----------------------------------------------------------------------------------------
    # Convert all inputs, labels, and attentions into torch tensors, the required datatype: torch.int64
    train_input_ids = torch.tensor(torch.from_numpy( np.array( list(annotations['embeddings']), dtype=np.float32) ), dtype=torch.int64)                    
    train_input_labels = torch.tensor(torch.from_numpy( np.array( list(annotations['label_pads']), dtype=np.int64) ), dtype=torch.int64)
    train_attn_masks = torch.tensor(torch.from_numpy( np.array( list(annotations['attn_masks']), dtype=np.int64) ), dtype=torch.bool)
    train_pos_tags = torch.nn.functional.one_hot( torch.tensor( torch.from_numpy( np.array( list(annotations['inputpos']), dtype=np.int64) ), dtype=torch.int64) )

    # dev_input_ids = torch.tensor(torch.from_numpy(X_dev), dtype=torch.int64)                    
    # dev_input_labels = torch.tensor(torch.from_numpy(y_dev), dtype=torch.int64)
    # dev_attn_masks = torch.tensor(torch.from_numpy(dev_masks), dtype=torch.bool)

    # Test set (EBM-NLP training data used as test set)
    test_input_ids = torch.tensor(torch.from_numpy( np.array( list(annotations_testdf_['embeddings']), dtype=np.float32) ), dtype=torch.int64)                    
    test_input_labels = torch.tensor(torch.from_numpy( np.array( list(annotations_testdf_['label_pads']), dtype=np.int64) ), dtype=torch.int64)
    test_attn_masks = torch.tensor(torch.from_numpy( np.array( list(annotations_testdf_['attn_masks']), dtype=np.int64) ), dtype=torch.bool)
    test_pos_tags = torch.nn.functional.one_hot( torch.tensor( torch.from_numpy( np.array( list(annotations_testdf_['inputpos']), dtype=np.int64) ), dtype=torch.int64) )

    # Test set 1 (EBM-NLP test gold data test set)
    test1_input_ids = torch.tensor(torch.from_numpy( np.array( list(annotations_test1df_['embeddings']), dtype=np.float32) ), dtype=torch.int64)                    
    test1_input_labels = torch.tensor(torch.from_numpy( np.array( list(annotations_test1df_['label_pads']), dtype=np.int64) ), dtype=torch.int64)
    test1_attn_masks = torch.tensor(torch.from_numpy( np.array( list(annotations_test1df_['attn_masks']), dtype=np.int64) ), dtype=torch.bool)
    test1_pos_tags = torch.nn.functional.one_hot( torch.tensor( torch.from_numpy( np.array( list(annotations_test1df_['inputpos']), dtype=np.int64) ), dtype=torch.int64) )

    # Test set 2 (Hilfiker test set)
    test2_input_ids = torch.tensor(torch.from_numpy( np.array( list(annotations_test2df_['embeddings']), dtype=np.float32) ), dtype=torch.int64)                    
    test2_input_labels = torch.tensor(torch.from_numpy( np.array( list(annotations_test2df_['label_pads']), dtype=np.int64) ), dtype=torch.int64)
    test2_attn_masks = torch.tensor(torch.from_numpy( np.array( list(annotations_test2df_['attn_masks']), dtype=np.int64) ), dtype=torch.bool)            
    test2_pos_tags = torch.nn.functional.one_hot( torch.tensor( torch.from_numpy( np.array( list(annotations_test2df_['inputpos']), dtype=np.int64) ), dtype=torch.int64) )
    print('Inputs converted to tensors...')

    # ----------------------------------------------------------------------------------------
    # Create dataloaders from the tensors
    # ----------------------------------------------------------------------------------------
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_input_ids, train_input_labels, train_attn_masks, train_pos_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=None, batch_size=20, shuffle=False)

    # Create the DataLoader for our development set. XXX Do not pass this internal training set dataloader for another experiment
    # dev_data = TensorDataset(dev_input_ids, dev_input_labels, dev_attn_masks)
    # dev_sampler = RandomSampler(dev_data)
    # development_dataloader = DataLoader(dev_data, sampler=None, batch_size=10, shuffle=False)

    # Create the DataLoader for our test set. (This will be used as validation set!)
    test_data = TensorDataset(test_input_ids, test_input_labels, test_attn_masks, test_pos_tags)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=None, batch_size=20, shuffle=False)

    # Create the DataLoader for our test set 1.
    test1_data = TensorDataset(test1_input_ids, test1_input_labels, test1_attn_masks, test1_pos_tags)
    test1_sampler = RandomSampler(test1_data)
    test1_dataloader = DataLoader(test1_data, sampler=None, batch_size=20, shuffle=False)

    # Create the DataLoader for our test set 2.
    test2_data = TensorDataset(test2_input_ids, test2_input_labels, test2_attn_masks, test2_pos_tags)
    test2_sampler = RandomSampler(test2_data)
    test2_dataloader = DataLoader(test2_data, sampler=None, batch_size=20, shuffle=False)

    print('\n--------------------------------------------------------------------------------------------')
    print('Data loaders created')
    print('--------------------------------------------------------------------------------------------')

    ##################################################################################
    # Parse the arguments for training
    ##################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type = int, default = device)
    parser.add_argument('-freeze_bert', action='store_false') # store_false = won't freeze BERT
    parser.add_argument('-print_every', type = int, default= 100)
    parser.add_argument('-max_eps', type = int, default= 10)
    parser.add_argument('-lr', type = float, default= 5e-4)
    parser.add_argument('-eps', type = float, default= 1e-8)
    parser.add_argument('-loss', type = str, default = 'general')
    # parser.add_argument('-fold', type = str, default=fold)
    parser.add_argument('-bidrec', type = str, default=True)
    args = parser.parse_args()

    # PATH_SUM_WRITER = '/home/anjani/DistantCTO/ModelTraining/logs/' + str(fold)
    PATH_SUM_WRITER = '/home/anjani/DistantCTO/ModelTraining/logs/' + str('noEBMTrain_w_shortAnnot_SCIPOSAtten_crf')
    writer = SummaryWriter(PATH_SUM_WRITER) # XXX

    ##################################################################################
    #Instantiating the BERT model
    ##################################################################################
    print("Building model...")
    st = time.time()
    model = choose_model(exp_args.embed, exp_args.model, args)

    ##################################################################################
    # Tell pytorch to run data on this model on the GPU and parallelize it
    ##################################################################################
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids = [0])
    #     print("Using", len(model.device_ids), " GPUs!")
    model.cuda()
    print("Done in {} seconds".format(time.time() - st))

    ##################################################################################
    st = time.time()
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    optimizer = AdamW(model.parameters(),
                    lr = args.lr, # args.learning_rate - default is 5e-5 (for BERT-base)
                    eps = args.eps, # args.adam_epsilon  - default is 1e-8.
                    )

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
    # train(model, optimizer, scheduler, train_dataloader, test_dataloader, args, exp_args)
    # train(model, optimizer, scheduler, train_dataloader, development_dataloader, args, exp_args)
    print("Training and validation done in {} seconds".format(time.time() - st))

    print('##################################################################################')
    print('Begin test...')
    print('##################################################################################')
    # checkpoint = torch.load(saved_models[-1], map_location='cuda:0')
    checkpoint = torch.load('/mnt/nas2/results/Results/systematicReview/Distant_CTO/models/intervention/combined_w_shortAnn_IO_POSciAtten_linear/bert_bilstm_crf_epoch_9_best_model.pth', map_location='cuda:0')
    model.load_state_dict( checkpoint )

    # # print('Applying the best model on test set (EBM-NLP training set used as test set)...')
    # # test_cr, all_pred_flat, all_GT_flat, cm = evaluate(model, optimizer, scheduler, test_dataloader, args, exp_args)
    # # print(test_cr)

    print('Applying the best model on test set (EBM-NLP)...')
    test1_cr, all_pred_flat, all_GT_flat, cm = evaluate(model, optimizer, scheduler, test1_dataloader, args, exp_args)
    print(test1_cr)
    print(cm)

    print('Applying the best model on test set (Hilfiker et al.)...')
    test2_cr, all_pred_flat, all_GT_flat, cm = evaluate(model, optimizer, scheduler, test2_dataloader, args, exp_args)
    print(test2_cr)
    print(cm)