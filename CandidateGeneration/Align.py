#!/usr/bin/env python
'''
This module contains functions to align source intervention terms to the target sentences (short targets, long targets - with and without negative candidates). It also has function to extract the annotations once source terms are aligned to the target.
'''
__author__ = "Anjani Dhrangadhariya"
__maintainer__ = "Anjani Dhrangadhariya"
__email__ = "anjani.k.dhrangadhariya@gmail.com"
__status__ = "Prototype/Research"

import sys, json, os
import logging
import datetime as dt
import time
import random

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search,  Q

import difflib, re
import nltk
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize

from collections import Counter
from collections import defaultdict
import collections
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *

from Preprocessing import *
from ExtractData import *
from Scoring import *

####################################################################
# Function to extract the aligned candidate annotations
####################################################################
def extractAnnotation(source, target, match):
    
    token = list()
    annot = list()
    
    span_generator = WhitespaceTokenizer().span_tokenize(target)
    # print( list(span_generator) )

    annotation_start_position = match[1][0]
    annotation_stop_position = match[1][0] + match[1][2]

    annotation = [0] * len(target)
    for n, i in enumerate(annotation):
        if n >= annotation_start_position and n <= annotation_stop_position:
            annotation[n] = 1

    for span in span_generator:
        token_ = target[span[0]:span[1]]

        annot_ = annotation[span[0]:span[1]]
        max_element_i = Counter(annot_)
        max_element = max_element_i.most_common(1)

        token.append(token_)
        annot.append(max_element)

    # Check if the number of annotations match number of tokens present in the sentence
    assert len(token) == len(annot)
        
    return token, annot

def extract(source, target_text, target, match):
    
    annot = []

    annotation_start_position = match[1][0]
    annotation_stop_position = match[1][0] + match[1][2]

    annot_init = '0' * len(target_text)
    match_annot = ['1' if i in range(annotation_start_position, annotation_stop_position) else '0' for i, a in enumerate(annot_init)]
    match_annot = ''.join(match_annot)
    
    for i, (s,t) in enumerate(zip(target['idx'], target['tokens'])):

        if i != len(target['idx'])-1:
            start_a = s
            end_a = target['idx'][i+1]   
        else:
            start_a = s
            end_a = len(target_text)

        annot_t = match_annot[start_a:end_a]

        if len(annot_t) != len(t):
            annot.append(max(annot_t))
        else:
            annot.append(max(annot_t))


    assert len(annot) == len(target['tokens'])
    target['annotation'] = annot

    return target

###############################################################################################
# Function to align source intervention terms with high confidence short targets
###############################################################################################
def align_highconf_target(target, source):
    annot = list() # Get's updated for each Intervention name identified
    token = list()
    extracted_annot = dict()

    target_term = target['text'].lower()

    if target_term is not None:
        # Match the source to the target
        s = difflib.SequenceMatcher(None, target_term, source, autojunk=True)
        matches = fullMatchScore(s, source, target_term)
        for match in matches:
            if match[0] >= 0.9:               
                extracted_annot = extract( source, target_term, target, match )

    if extracted_annot:
        return target['tokens'], extracted_annot['annotation']
    else:
        return token, annot


###############################################################################################
# Function to align source intervention terms with high confidence long targets
###############################################################################################
def align_highconf_longtarget(target, source):

    target_sentences = list()
   
    if target is not None :
        # Sentence tokenization
        target_sentences = sent_tokenize(target)
        collect_annotations = dict()
        
        # Iterate each sentence
        for i, eachSentence in enumerate(target_sentences):

            annot = list() # Get's updated for each Intervention name identified and for each sentence
            token = list()

            s = difflib.SequenceMatcher(None, eachSentence, source, autojunk=True)
            matches = fullMatchScore(s, source, target)
            match_scores = [item[0] for item in matches ]

            if 1.0 in match_scores:
                for match in matches:
                    if match[0] == 1.0:                     
                        token_i, annot_i = extractAnnotation(source, eachSentence, match)
                        annot.extend( annot_i )
                        token.extend( token_i )
                
            if annot:
                token_annot = [token, annot, [eachTuple[1]  for eachTuple in nltk.pos_tag_sents([token])[0]]]
                collect_annotations['sentence' + str(i)] = token_annot

    assert len(token) == len(annot)

    return collect_annotations

#######################################################################################################################################################
# Function to align source intervention terms with high confidence long targets containing both positive and negative annotated candidates
#######################################################################################################################################################
def align_highconf_longtarget_negSent(target, source):

    target_sentences = list()
   
    if target is not None :
        # Sentence tokenization
        target_sentences = sent_tokenize(target)
        collect_annotations = dict()
        
        # Check if each sentence has the source intervention in it
        for i, eachSentence in enumerate(target_sentences):

            annot = list() # Get's updated for each Intervention name identified and for each sentence
            token = list()

            s = difflib.SequenceMatcher(None, eachSentence, source, autojunk=True)
            matches = fullMatchScore(s, source, target)
            match_scores = [item[0] for item in matches ]

            if 1.0 in match_scores:
                for match in matches:
                    if match[0] == 1.0:                     
                        token_i, annot_i = extractAnnotation(source, eachSentence, match)
                        annot.extend( annot_i )
                        token.extend( token_i )

            if annot:
                token_annot = [token, annot, [eachTuple[1]  for eachTuple in nltk.pos_tag_sents([token])[0]]]
                # collect_annotations['sentence' + str(i)] = token_annot
                collect_annotations[str(i)] = token_annot

            if 1.0 not in match_scores and all( list(all(i <= 0.20 for i in match_scores)) ): # very negative sentences (possibility of false negative candidates)
                # print('VERY NEGATIVE sentence' + str(i))
                tokenized_negative_sentence_i = eachSentence.split(' ')
                annotation_negative_sentence_i = [0] * len(tokenized_negative_sentence_i)
                annot.extend( annotation_negative_sentence_i )
                token.extend( tokenized_negative_sentence_i )

            if annot:
                token_annot = [token, annot, [eachTuple[1]  for eachTuple in nltk.pos_tag_sents([token])[0]]]
                # collect_annotations['sentence' + str(i)] = token_annot
                collect_annotations[str(i)] = token_annot

    assert len(token) == len(annot)
    return collect_annotations