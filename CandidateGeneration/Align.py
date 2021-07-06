#!/usr/bin/env python
'''
This module contains functions to obtain block-wise match score and (summed) block-wise for the matched Source and Target texts.
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

def extractAnnotation(source, target, match):
    
    token = list()
    annot = list()
    
    span_generator = WhitespaceTokenizer().span_tokenize(target)

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
        max_element = max_element_i.most_common(1)[0][0]
        #annot_correct = [max_element] * len(annot_)

        token.append(token_)
        annot.append(max_element)

    # Check if the number of annotations match number of tokens present in the sentence
    assert len(token) == len(annot)
        
    return token, annot

def align_highconf_shorttarget(target, source):
    annot = list() # Get's updated for each Intervention name identified
    token = list()

    if target is not None:
        # Match the source to the target
        s = difflib.SequenceMatcher(None, target, source, autojunk=True)
        matches = fullMatchScore(s, source, target)
        for match in matches:
            if match[0] == 1.0:                    
                token, annot = extractAnnotation(source, target, match)

    assert len(token) == len(annot)
    return token, annot

# Collects annotations for all the sentences in the target and returns them
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
                        # print('MATCH 1 sentence' + str(i))

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