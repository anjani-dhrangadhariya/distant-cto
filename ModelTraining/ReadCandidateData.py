#!/usr/bin/env python
'''
Main python file for reading candidates generated during the candidate generation phase
'''
__author__ = "Anjani Dhrangadhariya"
__maintainer__ = "Anjani Dhrangadhariya"
__email__ = "anjani.k.dhrangadhariya@gmail.com"
__status__ = "Prototype/Research"

import logging
import datetime as dt
import datetime
import glob
import argparse
import pdb
import random
import sys, json, os
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from DataBuilders.SemanticVectorBuilder import *
from DataBuilders.ContextualVectorBuilder import *
from DataBuilders.ValidationDataBuilder import *
from DataBuilders.ValidationSentenceBuilder import *

def flatten(l):
    flatList = []
    for elem in l:
        # if an element of a list is a list
        # iterate over this list and add elements to flatList 
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList

def labelGenerator(raw_labels):

    bio_labels = []
    for i, eachLabel in enumerate(raw_labels):
        if eachLabel == 0:
            bio_labels.append(0)
        elif raw_labels[i] == 1 and raw_labels[i-1] == 0:
            bio_labels.append(1)
        elif raw_labels[i] == 1 and raw_labels[i-1] == 1:
            bio_labels.append(2)  
    return bio_labels

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

def readRawCandidates( list_NCT, extraction_number, label_type=None ):
    counter_stop = 0
    list_NCT_ids = []

    tokens = []
    labels = []
    pos = []
    nct_ids = []
    length_examiner = []
    with open(list_NCT, 'r', encoding='latin1') as NCT_ids_file:
        next(NCT_ids_file)
        for eachLine in NCT_ids_file:
            annot = json.loads(eachLine)
            id_ = annot['id']

            # Check if aggregate annotations are present in the json
            if 'aggregate_annot' in annot.keys():

                # Read official title
                if 'official_title_pos' in annot['aggregate_annot'].keys():
                    assert len(annot['aggregate_annot']['official_title']) == len(annot['aggregate_annot']['official_title_annot'])
                    raw_labels =  annot['aggregate_annot']['official_title_annot']
                    if label_type == 'BIO':
                       raw_labels = labelGenerator(raw_labels)
                    tokens.append( annot['aggregate_annot']['official_title'] )
                    labels.append( raw_labels )
                    pos.append( annot['aggregate_annot']['official_title_pos'] )
                    length_examiner.append( len(annot['aggregate_annot']['official_title_annot']) )
                    nct_ids.append(id_)

                if 'brief_title' in annot['aggregate_annot'].keys():
                    assert len(annot['aggregate_annot']['brief_title']) == len(annot['aggregate_annot']['brief_title_annot'])
                    raw_labels =  annot['aggregate_annot']['brief_title_annot']
                    if label_type == 'BIO':
                       raw_labels = labelGenerator(raw_labels)
                    tokens.append( annot['aggregate_annot']['brief_title'] )
                    labels.append( raw_labels )
                    pos.append( annot['aggregate_annot']['brief_title_pos'] )
                    length_examiner.append( len(annot['aggregate_annot']['brief_title_annot']) )
                    nct_ids.append(id_)

                if 'brief_summary_annot' in annot['aggregate_annot'].keys():
                    # iterate the dictionary
                    for eachKey, eachValue in annot['aggregate_annot']['brief_summary_annot'].items():
                        assert len(eachValue) == 3
                        raw_labels =  eachValue[1]
                        if label_type == 'BIO':
                            raw_labels = labelGenerator(raw_labels)
                        tokens.append( eachValue[0] )
                        labels.append( raw_labels )
                        pos.append( eachValue[2] )
                        length_examiner.append( len( eachValue[1] ) )
                        nct_ids.append(id_)

                if 'detailed_description_annot' in annot['aggregate_annot'].keys():
                    # iterate the dictionary
                    for eachKey, eachValue in annot['aggregate_annot']['detailed_description_annot'].items():
                        assert len(eachValue) == 3
                        raw_labels =  eachValue[1]
                        if label_type == 'BIO':
                            raw_labels = labelGenerator(raw_labels)
                        tokens.append( eachValue[0] )
                        labels.append( raw_labels )
                        pos.append( eachValue[2] )
                        length_examiner.append( len( eachValue[1] ) )
                        nct_ids.append(id_)

                # if 'intervention_description_annot' in annot['aggregate_annot'].keys():
                #     # iterate the dictionary
                #     for eachKey, eachAnnot in enumerate(annot['aggregate_annot']['intervention_description_annot']):
                #         raw_labels =  eachAnnot[1]
                #         print(eachAnnot)
                #         raw_tokens = eachAnnot[0]
                #         if label_type == 'BIO':
                #             raw_labels = labelGenerator(raw_labels)
                #         tokens.append( raw_tokens )
                #         labels.append( raw_labels)
                #         pos.append( eachValue[2] )
                #         length_examiner.append( len( eachValue[1] ) )
                #         nct_ids.append(id_)

                if 'intervention_description_annot' in annot['aggregate_annot'].keys():
                    # iterate the dictionary
                    for eachKey, eachValue in annot['aggregate_annot']['intervention_description_annot'].items():
                        assert len(eachValue) == 3
                        raw_labels =  eachValue[1]
                        if label_type == 'BIO':
                            raw_labels = labelGenerator(raw_labels)
                        tokens.append( eachValue[0] )
                        labels.append( raw_labels )
                        pos.append( eachValue[2] )
                        length_examiner.append( len( eachValue[1] ) )
                        nct_ids.append(id_)

            counter_stop = counter_stop + 1
            if counter_stop == 50:
                break

    # Collate the lists into a dataframe
    # corpusDf = percentile_list = pd.DataFrame({'tokens': tokens,'labels': labels,'ids': nct_ids})
    corpusDf = percentile_list = pd.DataFrame({'tokens': tokens,'labels': labels, 'pos': pos})
    
    df = corpusDf.sample(frac=1).reset_index(drop=True) # Shuffles the dataframe after creation

    MAX_LEN = max(length_examiner)

    return df, MAX_LEN


def FetchTrainingCandidates():
    
    # POS label encoder
    le_pos = preprocessing.LabelEncoder()

    # List of arguments to set up experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('-embed', type = str, default = 'scibert') # word2vec, bio2vec2, bio2vec30, bert, gpt2, biobert, scibert
    parser.add_argument('-embed_type', type = str, default = 'contextual') # semantic, contextual
    parser.add_argument('-model', type = str, default = 'scibertposattenlinear') # bertcrf, scibertcrf, scibertposcrf, scibertposlinear, scibertposattenlinear, scibertposattencrf
    parser.add_argument('-label_type', type = str, default = 'seq_lab') # seq_lab, BIO, BIOES
    parser.add_argument('-text_level', type = str, default = 'sentence') # sentence, document
    args = parser.parse_args()

    print('Chosen embedding type is: ', args.embed)

    # List of NCT ids to get annotations for... Training data
    list_NCT = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/intervention_data_preprocessed/extraction1_partofspeech.txt'
    print('Fetching raw annotations for the candidates...')
    annotations_df, MAX_LEN = readRawCandidates( list_NCT, str(1), args.label_type ) # str(1) = extraction 1
    print('Length of annotation corpus: ', len(annotations_df))

    EBM_NLP_texts = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/documents/'
    # testDf, test1Df, test2Df = get_data_loaders(EBM_NLP_texts, args.label_type)

    MAX_LEN = 100 # Check the average MAX_LEN of the annotations and sentences in the training and test sets

    # Validation and test data
    # convert tokens and labels into vectors
    if args.embed_type == 'semantic':
        # Semantic vectors fetched 
        print('Extracting semantic embeddings from the text tokens...')
        embeddings, labels, attention_masks = get_semantic_vectors(annotations_df, args.embed, MAX_LEN)

    elif args.embed_type == 'contextual':
        print('Extracting contextual embeddings from the text tokens...')
        if args.text_level == 'document':

            embeddings, labels, attention_masks = get_contextual_vectors(annotations_df, args.embed, MAX_LEN)

            embeddings_test, labels_test, attention_masks_test = get_contextual_vectors(testDf, args.embed, 510)
            embeddings_test1, labels_test1, attention_masks_test1 = get_contextual_vectors(test1Df, args.embed, 510)
            embeddings_test2, labels_test2, attention_masks_test2 = get_contextual_vectors(test2Df, args.embed, 510)

            assert len(annotations_df['labels']) == len(embeddings)
            assert len(annotations_df['tokens']) == len(embeddings)

            annotations_df_ = annotations_df.assign(embeddings = pd.Series(embeddings).values, label_pads = pd.Series(labels).values, attn_masks = pd.Series(attention_masks).values) # assign the padded embeddings to the dataframe
            annotations_testdf_ = testDf.assign(embeddings = pd.Series(embeddings_test).values, label_pads = pd.Series(labels_test).values, attn_masks = pd.Series(attention_masks_test).values) # assign the padded embeddings to the dataframe
            annotations_test1df_ = test1Df.assign(embeddings = pd.Series(embeddings_test1).values, label_pads = pd.Series(labels_test1).values, attn_masks = pd.Series(attention_masks_test1).values) # assign the padded embeddings to the dataframe
            annotations_test2df_ = test2Df.assign(embeddings = pd.Series(embeddings_test2).values, label_pads = pd.Series(labels_test2).values, attn_masks = pd.Series(attention_masks_test2).values) # assign the padded embeddings to the dataframe

            return annotations_df_, annotations_testdf_, annotations_test1df_, annotations_test2df_, args

        elif args.text_level == 'sentence':

            print('Length of annotation corpus: ', len(annotations_df))

            EBM_NLP_sentences = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/sentence_annotation2POS.txt'
            testDf = readEBMNLP_sentAnnot(EBM_NLP_sentences, args.label_type)

            EBM_NLPgold_sentences = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebmnlpgold_sentence_annotation2POS.txt'
            test1Df = readEBMNLP_sentAnnot(EBM_NLPgold_sentences, args.label_type)
            print('Loading EBM-NLP training and test sentences completed...')

            hilfiker_sentences = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/hilfiker_sentence_annotation2POS.txt'
            test2Df = readEBMNLP_sentAnnot(hilfiker_sentences, args.label_type)
            print('Loading in-house physiotherapy test sentences completed...')

            fullPOS_ = flatten( annotations_df['pos'].values.tolist() )
            fullPOS__ = flatten( testDf['pos'].values.tolist() )
            fullPOS___ = flatten( test1Df['pos'].values.tolist() )
            fullPOS____ = flatten( test1Df['pos'].values.tolist() )
            fullPOS = fullPOS_ + fullPOS__ + fullPOS___ + fullPOS____
            fullPOS = list(set(fullPOS))
            pos_encoded = le_pos.fit_transform( fullPOS )

            # print('executing for the weak corpus....')
            embeddings, labels, attention_masks, input_pos = get_contextual_vectors(annotations_df, args.embed, le_pos, MAX_LEN)
            # print('executing for the EBMNLP training corpus....')
            embeddings_test, labels_test, attention_masks_test, input_pos_test  = get_contextual_vectors(testDf, args.embed, le_pos, MAX_LEN)
            # print('executing for the EBMNLP test corpus....')
            embeddings_test1, labels_test1, attention_masks_test1, input_pos_test1  = get_contextual_vectors(test1Df, args.embed, le_pos, MAX_LEN)
            # print('executing for the hilfiker test corpus....')
            embeddings_test2, labels_test2, attention_masks_test2, input_pos_test2  = get_contextual_vectors(test2Df, args.embed, le_pos, MAX_LEN)
            assert len(annotations_df['labels']) == len(embeddings)
            assert len(annotations_df['tokens']) == len(embeddings)
            
            annotations_df_ = annotations_df.assign(embeddings = pd.Series(embeddings).values, label_pads = pd.Series(labels).values, attn_masks = pd.Series(attention_masks).values, inputpos = pd.Series(input_pos).values) # assign the padded embeddings to the dataframe
            annotations_testdf_ = testDf.assign(embeddings = pd.Series(embeddings_test).values, label_pads = pd.Series(labels_test).values, attn_masks = pd.Series(attention_masks_test).values, inputpos = pd.Series(input_pos_test).values) # assign the padded embeddings to the dataframe
            annotations_test1df_ = test1Df.assign(embeddings = pd.Series(embeddings_test1).values, label_pads = pd.Series(labels_test1).values, attn_masks = pd.Series(attention_masks_test1).values, inputpos = pd.Series(input_pos_test1).values) # assign the padded embeddings to the dataframe
            annotations_test2df_ = test2Df.assign(embeddings = pd.Series(embeddings_test2).values, label_pads = pd.Series(labels_test2).values, attn_masks = pd.Series(attention_masks_test2).values, inputpos = pd.Series(input_pos_test2).values) # assign the padded embeddings to the dataframe

            return annotations_df_, annotations_testdf_, annotations_test1df_, annotations_test2df_, args