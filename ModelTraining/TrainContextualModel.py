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
from collections import OrderedDict

# Memory leak
import gc

# statistics
import statistics

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
from sklearn.metrics import plot_confusion_matrix
from tensorboardX import SummaryWriter

from ReadCandidateData import *
from Models.semantic_crf import SemanticCRF
from Models.ChooseModel import choose_model
from Functions.ThresholdMoving import getRoc

from pandas import DataFrame

##################################################################################
# set up the GPU
##################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print('Number of GPUs identified: ', n_gpu)
print('You are using ', torch.cuda.get_device_name(0), ' : ', device , ' device')
# print('You are using ', torch.cuda.get_device_name(1), ' : ', device , ' device')

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

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

# get logits

def getLogits(all_preds_tens):
    # get logits from predictions
    max_probs = torch.max(all_preds_tens, dim=2) # get the highest of two probablities
    all_logits_tens = max_probs.indices.flatten()
    return all_logits_tens

# Evaluate
def evaluate(defModel, optimizer, scheduler, development_dataloader, args, exp_args, epoch_number = None):
    mean_acc = 0
    mean_loss = 0
    count = 0
    total_val_loss_coarse = 0

    class_rep_temp = []

    with torch.no_grad() :

        # collect all the evaluation predictions and ground truth here
        all_GT = []
        all_masks = []
        all_predictions = []
        all_tokens = []

        for e_input_ids_, e_labels, e_input_mask, e_input_pos in development_dataloader:

            e_input_ids_ = e_input_ids_.to(f'cuda:0')

            with torch.cuda.device_of(e_input_ids_.data):
                e_input_ids = e_input_ids_.clone()

            # check if the model is GPT2-based
            if 'gpt2' in exp_args.embed:
                e_input_ids[e_input_ids_==0] = tokenizer.unk_token_id
                e_input_ids[e_input_ids_==101] = tokenizer.bos_token_id
                e_input_ids[e_input_ids_==102] = tokenizer.eos_token_id

            # coarse grained entity labels
            e_input_mask = e_input_mask.to(f'cuda:0')
            e_labels = e_labels.to(f'cuda:0')
            e_input_pos = e_input_pos.to(f'cuda:0')

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

            # mean_loss += e_loss.item()
            mean_loss += abs( torch.mean(e_loss) ) 

            for i in range(0, e_labels.shape[0]):

                # mask the probas
                masked_preds = torch.masked_select( e_logits[i, ].to(f'cuda:0'), e_mask[i, ] )
                # mask the labels
                masked_labs = torch.masked_select( e_labels[i, ].to(f'cuda:0'), e_mask[i, ] )
                temp_cr = classification_report(y_pred= masked_preds.cpu(), y_true=masked_labs.cpu(), labels=list(range(2)), output_dict=True) 
                class_rep_temp.append(temp_cr['macro avg']['f1-score'])

                all_masks.extend( e_mask[i, ] )
                all_GT.extend( e_labels[i, ] )
                all_predictions.extend( e_logits[i, ] )
                all_tokens.extend( e_input_ids[i, ] )

            count += 1

        avg_val_loss = mean_loss / len(development_dataloader)     
        # writer.add_scalar('loss-validation', avg_val_loss, epoch_number) # XXX Return the avg_val_loss from this method to the train method and write to the writer from there

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
        # getRelInputs( all_token_tens , defModel )
        selected_tokens_coarse = torch.masked_select( all_token_tens.to(f'cuda:0'), all_masks_tens )

        # flatten the masked tensors
        all_pred_flat = np.asarray(selected_preds_coarse.cpu(), dtype=np.float32).flatten()
        all_GT_flat = np.asarray(selected_labs_coarse.cpu(), dtype=np.float32).flatten()
        all_tokens_flat = np.asarray(selected_tokens_coarse.cpu(), dtype=np.int64).flatten()

        # Final classification report and confusion matrix for each epoch
        val_cr = classification_report(y_pred= all_pred_flat, y_true=all_GT_flat, labels=list(range(2)), output_dict=True)     

        # confusion_matrix and plot
        labels = [1,0]
        cm = confusion_matrix(all_GT_flat, all_pred_flat, labels, normalize=None)

    return val_cr, all_pred_flat, all_GT_flat, cm, all_tokens_flat, class_rep_temp


# Train
def train(defModel, optimizer, scheduler, train_dataloader, development_dataloader, args, exp_args, eachSeed):

    with torch.enable_grad():
        best_meanf1 = 0.0
        for epoch_i in range(0, args.max_eps):
        
            # Accumulate loss over an epoch
            total_train_loss = 0

            # (coarse-grained) accumulate predictions and labels over the epoch
            train_epoch_logits_coarse_i = []
            train_epochs_labels_coarse_i = []

            # Training for all the batches in this epoch
            for step, batch in enumerate(train_dataloader):

                # Clear the gradients
                optimizer.zero_grad()

                b_input_ids = batch[0].to(f'cuda:{model.device_ids[0]}')
                b_labels = batch[1].to(f'cuda:{model.device_ids[0]}')
                b_masks = batch[2].to(f'cuda:{model.device_ids[0]}')
                b_pos = batch[3].to(f'cuda:{model.device_ids[0]}')

                b_loss, b_output, b_labels, b_mask = defModel(input_ids = b_input_ids, attention_mask=b_masks, labels=b_labels, input_pos=b_pos)
                
                # total_train_loss += abs( b_loss.item() )
                total_train_loss += abs( torch.mean(b_loss) ) 

                # abs( torch.FloatTensor( [statistics.mean(b_loss.tolist())] ) ).backward()
                abs( torch.mean(b_loss) ).backward()

                # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(defModel.parameters(), 1.0)

                #Optimization step
                optimizer.step()

                # Update the learning rate.
                scheduler.step()


                if len(list( b_output.shape )) > 1:
                    if 'crf' not in exp_args.model:
                        max_probs = torch.max(b_output, dim=2) # get the highest of two probablities
                        b_logits = max_probs.indices
                    else:
                        b_logits = b_output
                else: 
                    b_logits = b_output

                for i in range(0, b_labels.shape[0]):

                    selected_preds_coarse = torch.masked_select( b_logits[i, ].to(f'cuda:{model.device_ids[0]}'), b_mask[i, ])
                    selected_labs_coarse = torch.masked_select(b_labels[i, ].to(f'cuda:{model.device_ids[0]}'), b_mask[i, ])

                    train_epoch_logits_coarse_i.extend( selected_preds_coarse.to("cpu").numpy() )
                    train_epochs_labels_coarse_i.extend( selected_labs_coarse.to("cpu").numpy() )

                if step % args.print_every == 0:

                    cr = classification_report(y_pred=train_epoch_logits_coarse_i, y_true=train_epochs_labels_coarse_i, labels= list(range(2)), output_dict=True)
                    # meanF1_2 = cr['2']['f1-score']
                    meanF1_1 = cr['1']['f1-score']
                    meanF1_0 = cr['0']['f1-score']
                    # mean_1 = (meanF1_1 + meanF1_2) / 2
                    print('Training: Epoch {} with mean F1 score (1): {}, mean F1 score (0): {}'.format(epoch_i, meanF1_1 , meanF1_0))

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            # writer.add_scalar('loss-train', avg_train_loss, epoch_i)

            train_cr = classification_report(y_pred= train_epoch_logits_coarse_i, y_true=train_epochs_labels_coarse_i, labels= list(range(2)), output_dict=True) 
            # meanF1_2 = train_cr['2']['f1-score']
            meanF1_1 = train_cr['1']['f1-score']
            meanF1_0 = train_cr['0']['f1-score']
            mean_1 = (meanF1_1 + meanF1_0) / 2
            # writer.add_scalar('f1-train', meanF1_1  , epoch_i)
            
            # Delete the collected logits and labels
            del train_epoch_logits_coarse_i, train_epochs_labels_coarse_i
            gc.collect()

            # Validation 
            val_cr, all_pred_flat_coarse, all_GT_flat_coarse, cm, all_tokens_flat, class_rep_temp  = evaluate(defModel, optimizer, scheduler, development_dataloader, args, exp_args, epoch_i)
            # val_meanF1_2 = val_cr['2']['f1-score']
            val_meanF1_1 = val_cr['1']['f1-score']
            val_meanF1_0 = val_cr['0']['f1-score']
            val_mean_1 = (val_meanF1_1 + val_meanF1_0) / 2
            # print('Validation F1 for class intervention ', val_meanF1)
            print('Validation: Epoch {} with mean F1 score (1): {}, mean F1 score (0): {} and traing loss : {}'.format(epoch_i, val_meanF1_1, val_meanF1_0, avg_train_loss))
            # writer.add_scalar('f1-validation', val_meanF1_1 , epoch_i)

            if val_meanF1_1 > best_meanf1:
                print("Best validation mean F1 improved from {} to {} ...".format( best_meanf1, val_meanF1_1 ))
                model_name_here = '/mnt/nas2/results/Results/systematicReview/Distant_CTO/models/intervention/0_EBM_baseline7BiLSTMmasked/' + str(eachSeed) + '/bert_bilstm_crf_epoch_' + str(epoch_i) + '_best_model.pth'
                print('Saving the best model for epoch {} with mean F1 score of {} '.format(epoch_i, val_meanF1_1 )) 
                torch.save(defModel.state_dict(), model_name_here)
                saved_models.append(model_name_here)                     
                best_meanf1 = val_meanF1_1


##########################################################################
# define parameter space here
##########################################################################
saved_models = []

if __name__ == "__main__":

    for eachSeed in [ 0, 1, 42 ]:

        def seed_everything( seed ):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed_val)
            torch.backends.cudnn.deterministic = True
        seed_everything(eachSeed)
        print('The random seed is set to: ', eachSeed)

        # Week annotations, Crowd annotations from EBM-NLP, Gold annotations from medical professionals, Annotations from physiotherapy articles
        # annotations = week, annotations_testdf_ = EBNLP_training, annotations_test1df_ = EBMNLP_test, annotations_test2df_ = hilfiker
        annotations, annotations_testdf_, annotations_test1df_, annotations_test2df_, exp_args = FetchTrainingCandidates()

        # Combine the training (annotations) and development (annotations_testdf_) dataframes
        if exp_args.train_data == 'combined':
            fulldf = annotations.append(annotations_testdf_, ignore_index=True)
        elif exp_args.train_data == 'distant-cto': 
            fulldf = annotations
        elif exp_args.train_data == 'ebm': 
            fulldf = annotations_testdf_

        # shuffle the dataset and divide into training and validation sets
        fulldf = fulldf.sample(frac=1).reset_index(drop=True)
        annotations, annotations_testdf_ = train_test_split(fulldf, test_size=0.2) 
        print('Size of training set: ', len(annotations.index))
        print('Size of development set: ', len(annotations_testdf_.index))
        
        del fulldf
        gc.collect()
        # ----------------------------------------------------------------------------------------
        # Load the data into tensors
        # ----------------------------------------------------------------------------------------
        # Convert all inputs, labels, and attentions into torch tensors, the required datatype: torch.int64
        train_input_ids = torch.from_numpy( np.array( list(annotations['embeddings']), dtype=np.int64 )).clone().detach()
        train_input_labels = torch.from_numpy( np.array( list(annotations['label_pads']), dtype=np.int64) ).clone().detach()
        train_attn_masks = torch.from_numpy( np.array( list(annotations['attn_masks']), dtype=np.int64) ).clone().detach()
        train_pos_tags = torch.nn.functional.one_hot( torch.from_numpy( np.array( list(annotations['inputpos']), dtype=np.int64) ).clone().detach() )

        # Test set (EBM-NLP training data used as test set)
        test_input_ids = torch.from_numpy( np.array( list(annotations_testdf_['embeddings']), dtype=np.int64) ).clone().detach()
        test_input_labels = torch.from_numpy( np.array( list(annotations_testdf_['label_pads']), dtype=np.int64) ).clone().detach()
        test_attn_masks = torch.from_numpy( np.array( list(annotations_testdf_['attn_masks']), dtype=np.int64) ).clone().detach()
        test_pos_tags = torch.nn.functional.one_hot( torch.from_numpy( np.array( list(annotations_testdf_['inputpos']), dtype=np.int64) ).clone().detach() )

        # Test set 1 (EBM-NLP test gold data test set)
        test1_input_ids = torch.from_numpy( np.array( list(annotations_test1df_['embeddings']), dtype=np.int64) ).clone().detach() 
        test1_input_labels = torch.from_numpy( np.array( list(annotations_test1df_['label_pads']), dtype=np.int64) ).clone().detach() 
        test1_attn_masks = torch.from_numpy( np.array( list(annotations_test1df_['attn_masks']), dtype=np.int64) ).clone().detach()
        test1_pos_tags = torch.nn.functional.one_hot( torch.from_numpy( np.array( list(annotations_test1df_['inputpos']), dtype=np.int64) ).clone().detach() )

        # Test set 2 (Hilfiker test set)
        test2_input_ids = torch.from_numpy( np.array( list(annotations_test2df_['embeddings']), dtype=np.int64) ).clone().detach()                     
        test2_input_labels = torch.from_numpy( np.array( list(annotations_test2df_['label_pads']), dtype=np.int64) ).clone().detach()
        test2_attn_masks = torch.from_numpy( np.array( list(annotations_test2df_['attn_masks']), dtype=np.int64) ).clone().detach()           
        test2_pos_tags = torch.nn.functional.one_hot( torch.from_numpy( np.array( list(annotations_test2df_['inputpos']), dtype=np.int64) ).clone().detach() )
        print('Inputs converted to tensors...')

        # Delete the large dataframes here
        del annotations, annotations_testdf_, annotations_test1df_, annotations_test2df_
        gc.collect()

        # ----------------------------------------------------------------------------------------
        # Create dataloaders from the tensors
        # ----------------------------------------------------------------------------------------
        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_input_ids, train_input_labels, train_attn_masks, train_pos_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=None, batch_size=10, shuffle=False)

        del train_input_ids, train_input_labels, train_attn_masks, train_pos_tags, train_sampler
        gc.collect()

        # Create the DataLoader for our test set. (This will be used as validation set!)
        test_data = TensorDataset(test_input_ids, test_input_labels, test_attn_masks, test_pos_tags)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=None, batch_size=6, shuffle=False)

        del test_input_ids, test_input_labels, test_attn_masks, test_pos_tags, test_sampler
        gc.collect()

        # Create the DataLoader for our test set 1.
        test1_data = TensorDataset(test1_input_ids, test1_input_labels, test1_attn_masks, test1_pos_tags)
        test1_sampler = RandomSampler(test1_data)
        test1_dataloader = DataLoader(test1_data, sampler=None, batch_size=6, shuffle=False)

        del test1_input_ids, test1_input_labels, test1_attn_masks, test1_pos_tags, test1_sampler
        gc.collect()

        # Create the DataLoader for our test set 2.
        test2_data = TensorDataset(test2_input_ids, test2_input_labels, test2_attn_masks, test2_pos_tags)
        test2_sampler = RandomSampler(test2_data)
        test2_dataloader = DataLoader(test2_data, sampler=None, batch_size=6, shuffle=False)

        del test2_input_ids, test2_input_labels, test2_attn_masks, test2_pos_tags, test2_sampler
        gc.collect()

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
        parser.add_argument('-bidrec', type = str, default=True)
        args = parser.parse_args()

        # PATH_SUM_WRITER = '/home/anjani/DistantCTO/ModelTraining/logs/' + str(fold)
        PATH_SUM_WRITER = '/home/anjani/DistantCTO/ModelTraining/logs/' + str('noEBMTrain_posnegtrail_SCIPOSAtten_crf')
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
        print('Number of devices used: ', torch.cuda.device_count())
        if exp_args.parallel == 'true':
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
                print("Using", len(model.device_ids), " GPUs!")
                print("Using", str(model.device_ids), " GPUs!")
            # model.cuda()
            model.to(f'cuda:{model.device_ids[0]}')
        elif exp_args.parallel == 'false':
            model = nn.DataParallel(model, device_ids = [0])
            # model.to(f'cuda:{model.device_ids[0]}')
            # model.cuda()

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
        train(model, optimizer, scheduler, train_dataloader, test_dataloader, args, exp_args, eachSeed)
        print("Training and validation done in {} seconds".format(time.time() - st))

        print('##################################################################################')
        print('Begin test...')
        print('##################################################################################')
        # Load the last saved BEST model
        checkpoint = torch.load(saved_models[-1], map_location='cuda:0')
        # or load a from some checkpoint
        # checkpoint = torch.load('/mnt/nas2/results/Results/systematicReview/Distant_CTO/models/intervention/0_EBM_baseline7masked/1/bert_bilstm_crf_epoch_7_best_model.pth', map_location='cuda:0')

        # Load the checkpoint/state dict into the predefined model and parallelize
        model.load_state_dict( checkpoint )
        if exp_args.parallel == 'true':
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
            # model.to(f'cuda:{model.device_ids[0]}')
        elif exp_args.parallel == 'false':
            # model = nn.DataParallel(model, device_ids = [0])
            model = torch.nn.DataParallel(model, device_ids=[0])
            # model.to(f'cuda:{model.device_ids[0]}')

        # # print('Applying the best model on test set (EBM-NLP training set used as test set)...')
        # # test_cr, all_pred_flat, all_GT_flat, cm = evaluate(model, optimizer, scheduler, test_dataloader, args, exp_args)
        # # print(test_cr)

        print('Applying the best model on test set (EBM-NLP)...')
        test1_cr, all_pred_flat, all_GT_flat, cm, test1_words, class_report_temp = evaluate(model, optimizer, scheduler, test1_dataloader, args, exp_args)
        print(test1_cr)
        print(cm)

        # # Write F1-scores for predictions on individual samples from the test datset for the t-test
        # df = DataFrame(class_report_temp)
        # with open('/home/anjani/distant-cto/ModelTraining/SignificanceTests/data/ebm/baseline.txt', 'a+') as writeFile:
        #     for index, row in df.iterrows():
        #         # print(str(row.values[0]))
        #         writeFile.write(str(row.values[0]))
        #         writeFile.write('\n')

        # tokens = []
        # for eachId in test1_words:
        #     token = model.tokenizer.convert_ids_to_tokens( [eachId] )
        #     tokens.append( token[0] )

        # dataframe_set1 = pd.DataFrame( {'tokens': tokens, 'predictions': all_pred_flat.tolist(), 'ground truth': all_GT_flat.tolist()} )
        # dataframe_set1.to_csv('/mnt/nas2/results/Results/systematicReview/Distant_CTO/predictions/ebm_test.csv', sep='\t', encoding='utf-8')

        print('Applying the best model on test set (Hilfiker et al.)...')
        test2_cr, all_pred_flat, all_GT_flat, cm, test2_words, class_report_temp = evaluate(model, optimizer, scheduler, test2_dataloader, args, exp_args)
        print(test2_cr)
        print(cm)

        # # Write F1-scores for predictions on individual samples from the test datset for the t-test
        # df = DataFrame(class_report_temp)
        # with open('/home/anjani/distant-cto/ModelTraining/SignificanceTests/data/physio/baseline.txt', 'a+') as writeFile:
        #     for index, row in df.iterrows():
        #         # print(str(row.values[0]))
        #         writeFile.write(str(row.values[0]))
        #         writeFile.write('\n')

        # tokens_2 = []
        # for eachId in test2_words:
        #     token = model.tokenizer.convert_ids_to_tokens( [eachId] )
        #     tokens_2.append( token[0] )

        # dataframe_set2 = pd.DataFrame( {'tokens': tokens_2, 'predictions': all_pred_flat.tolist(), 'ground truth': all_GT_flat.tolist()} )
        # dataframe_set2.to_csv('/mnt/nas2/results/Results/systematicReview/Distant_CTO/predictions/hilfiker.csv', sep='\t', encoding='utf-8')