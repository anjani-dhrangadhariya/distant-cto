import argparse
import datetime
import datetime as dt
import gc
import glob
import json
import logging
import os
import pdb
import random
from statistics import mean
import statistics
import sys
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from collections import Counter

from Utilities.experiment_arguments import *
from DataBuilders.ContextualVectorBuilder import *

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

# @profile
def readRawCandidates( list_NCT, annotation ):

    nct_ids = []
    tokens = []
    labels = []
    pos = []
    lengths = []

    with open(list_NCT, 'r', encoding='latin1') as NCT_ids_file:

        for i, eachLine in enumerate(NCT_ids_file):
            if len(eachLine) > 4:
                annot = json.loads(eachLine)
                id_ = annot['id']

                for target_key, target in annot.items():

                    if 'id' not in target_key and annotation in target and len(target['tokens']) >= 5: #and 'ds_np_fuzz' in target 

                        nct_ids.append( id_ )
                        tokens.append( target['tokens'] )
                        labels.append( target[annotation] )
                        pos.append( target['pos'] )
                        lengths.append( len(target['tokens']) )

                #if i == 20:
                #    break

    corpus_df = pd.DataFrame(
        {'ids': nct_ids,
        'tokens': tokens,
        'labels': labels,
        'pos': pos
        })

    #Elements_with_frequency = Counter(lengths)
    #maxlen = Elements_with_frequency.most_common()
    #maxlen = statistics.mean( lengths )
    #maxlen = max( lengths )
    maxlen = 100
    print(' The max length for sequences in the raw candidates is: ', maxlen)

    return corpus_df, maxlen

def readManuallyAnnoted( input_file_path ):

    nct_ids = []
    tokens = []
    labels = []
    pos = []

    with open(input_file_path, 'r', encoding='latin1') as NCT_ids_file:

        for i, eachLine in enumerate(NCT_ids_file):
            annot = json.loads(eachLine)

            for doc_key, document_annotations in annot.items():

                nct_ids.append(doc_key)
                tokens.append(document_annotations[0])
                labels.append(document_annotations[1])
                # TODO: Generate dummy POS items
                pos_i = ['NOUN'] * len( document_annotations[0] ) # TODO: Dummy POS
                pos.append( pos_i )

    corpus_df = pd.DataFrame(
        {'ids': nct_ids,
        'tokens': tokens,
        'labels': labels,
        'pos': pos
        })

    return corpus_df


def fetchAndTransformCandidates():

    # POS label encoder
    le_pos = preprocessing.LabelEncoder()

    args = getArguments() # get all the experimental arguments

    start_candidate_reading = time.time()
    raw_candidates, MAX_LEN = readRawCandidates( args.rawcand_file, annotation=args.annot )
    print("--- Took %s seconds to read the raw weakly annotated candidates ---" % (time.time() - start_candidate_reading))

    if args.train_data == 'distant-cto':
        maxlen = MAX_LEN
    if args.train_data == 'ebm-pico':
        maxlen = args.max_len
    if args.train_data == 'combined':
        maxlen = MAX_LEN

    # Retrieve EBM-PICO dataset here
    start_manual_reading = time.time()
    ebm_nlp = readManuallyAnnoted( args.ebm_nlp )
    ebm_gold = readManuallyAnnoted( args.ebm_gold )
    hilfiker = readManuallyAnnoted( args.hilfiker )
    print("--- Took %s seconds to read the manually annotated datasets ---" % (time.time() - start_manual_reading))

    fullPOS_ = flatten( raw_candidates['pos'].values.tolist() )
    fullPOS__ = flatten( ebm_nlp['pos'].values.tolist() )
    fullPOS___ = flatten( ebm_gold['pos'].values.tolist() )
    fullPOS____ = flatten( hilfiker['pos'].values.tolist() )
    fullPOS = fullPOS_ + fullPOS__ + fullPOS___ + fullPOS____
    fullPOS = list(set(fullPOS))
    pos_encoded = le_pos.fit_transform( fullPOS )

    start_candidate_transformation = time.time()
    tokenizer, model = choose_tokenizer_type( args.embed )
    input_embeddings, input_labels, input_masks, input_pos, tokenizer = getContextualVectors( raw_candidates, tokenizer, args.embed, maxlen, le_pos )
    assert len( input_embeddings ) == len( raw_candidates )
    print("--- Took %s seconds to transform the raw weakly annotated candidates ---" % (time.time() - start_candidate_transformation))

    start_manual_transformation = time.time()
    ebm_nlp_embeddings, ebm_nlp_labels, ebm_nlp_masks, ebm_nlp_pos, tokenizer = getContextualVectors( ebm_nlp, tokenizer, args.embed, maxlen, le_pos )
    ebm_gold_embeddings, ebm_gold_labels, ebm_gold_masks, ebm_gold_pos, tokenizer = getContextualVectors( ebm_gold, tokenizer, args.embed, maxlen, le_pos )
    hilfiker_embeddings, hilfiker_labels, hilfiker_masks, hilfiker_pos, tokenizer = getContextualVectors( hilfiker, tokenizer, args.embed, maxlen, le_pos )
    print("--- Took %s seconds to transform the manually annotated datasets ---" % (time.time() - start_manual_transformation))

    candidates_df = raw_candidates.assign(embeddings = pd.Series(input_embeddings).values, label_pads = pd.Series(input_labels).values, attn_masks = pd.Series(input_masks).values, inputpos = pd.Series(input_pos).values) # assign the padded embeddings to the dataframe
    ebm_nlp_df = ebm_nlp.assign(embeddings = pd.Series(ebm_nlp_embeddings).values, label_pads = pd.Series(ebm_nlp_labels).values, attn_masks = pd.Series(ebm_nlp_masks).values, inputpos = pd.Series(ebm_nlp_pos).values)
    ebm_gold_df = ebm_gold.assign(embeddings = pd.Series(ebm_gold_embeddings).values, label_pads = pd.Series(ebm_gold_labels).values, attn_masks = pd.Series(ebm_gold_masks).values, inputpos = pd.Series(ebm_gold_pos).values)
    hilfiker_df = hilfiker.assign(embeddings = pd.Series(hilfiker_embeddings).values, label_pads = pd.Series(hilfiker_labels).values, attn_masks = pd.Series(hilfiker_masks).values, inputpos = pd.Series(hilfiker_pos).values)

    del input_embeddings, input_labels, input_masks, input_pos, ebm_nlp_embeddings, ebm_nlp_labels, ebm_nlp_masks, ebm_nlp_pos, ebm_gold_embeddings, ebm_gold_labels, ebm_gold_masks, ebm_gold_pos, hilfiker_embeddings, hilfiker_labels, hilfiker_masks, hilfiker_pos, raw_candidates, ebm_nlp, ebm_gold, hilfiker  # Delete the large variables
    gc.collect()

    return candidates_df, ebm_nlp_df, ebm_gold_df, hilfiker_df, args, tokenizer, model