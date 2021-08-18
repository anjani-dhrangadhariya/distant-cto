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

from operator import add

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
from ExtractAnnot import *
from SourceTargets import *

def align_highconf_shorttarget(target, source):
    annot = list() # Get's updated for each Intervention name identified
    token = list()
    token_i = list()
    annot_i = list()

    if target is not None:
        # Match the source to the target
        s = difflib.SequenceMatcher(None, target, source, autojunk=True)
        matches = fullMatchScore(s, source, target)
        for match in matches:
            if match[0] == 1.0:                    
                token, annot = extract1Annotation(source, target, match)
            # if match[0] >= 0.9 and match[0] < 1.0:
            #     token_i, annot_i = extract09Annotation(source, target, match)

    assert len(token) == len(annot)
    assert len(token_i) == len(annot_i)

    if annot or annot_i:
        if annot:
            return token, annot
        if annot_i:
            return token, annot_i
    elif annot and annot_i:
        annot_added = list( map(add, annot, annot_i) )
        return token, annot_added

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

            if 1.0 in match_scores or True in list(all(i >= 0.9 and i < 1.0 for i in match_scores)):
                for match in matches:
                    if match[0] == 1.0:         
                        token_i, annot_i = extract1Annotation(source, eachSentence, match)
                        annot.extend( annot_i )
                        token.extend( token_i )
                    # if match[0] >= 0.9 and match[0] < 1.0:
                    #     token_i, annot_i = extract09Annotation(source, eachSentence, match)
                    #     annot.extend( annot_i )
                    #     token.extend( token_i )
                
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

            if 1.0 in match_scores or True in list(all(i >= 0.9 and i < 1.0 for i in match_scores)):
                for match in matches:
                    if match[0] == 1.0:         
                        token_i, annot_i = extract1Annotation(source, eachSentence, match)
                        annot.extend( annot_i )
                        token.extend( token_i )
                    # if match[0] >= 0.9 and match[0] < 1.0:
                    #     token_i, annot_i = extract09Annotation(source, eachSentence, match)
                    #     annot.extend( annot_i )
                    #     token.extend( token_i )

            if annot:
                token_annot = [token, annot, [eachTuple[1]  for eachTuple in nltk.pos_tag_sents([token])[0]]]
                # collect_annotations['sentence' + str(i)] = token_annot
                collect_annotations[str(i)] = token_annot

            if 1.0 not in match_scores and all( list(all(i <= 0.20 for i in match_scores)) ): # very negative sentences (possibility of false negative candidates)
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