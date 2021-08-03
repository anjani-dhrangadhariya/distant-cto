#!/usr/bin/env python
'''
This module contains functions to extract distantly-supervised annotations for confidence 1.0 and 0.9.
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


def extract1Annotation(source, target, match):
        
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

        if len(set(annot_)) > 1 and len(annot_) <= 2:
            max_element = 0
        else:
            max_element_i = Counter(annot_)
            max_element = max_element_i.most_common(1)[0][0]

        token.append(token_)
        annot.append(max_element)

    # Check if the number of annotations match number of tokens present in the sentence
    assert len(token) == len(annot)
        
    return token, annot


def extract09Annotation(source, target, match):
      
    token = list()
    annot = list()
    
    span_generator = WhitespaceTokenizer().span_tokenize(target)

    annotation_start_position = match[1][0]
    annotation_stop_position = match[1][0] + match[1][2] # example: Match(a=171, b=2, size=23)

    annotation = [0] * len(target)
    for n, i in enumerate(annotation):
        if n >= annotation_start_position and n <= annotation_stop_position:
            annotation[n] = 1

    for span in span_generator:

        token_ = target[span[0]:span[1]]
        annot_ = annotation[span[0]:span[1]]
        
        if len(set(annot_)) > 1 and len(annot_) <= 2:
            max_element = 0
        else:
            max_element_i = Counter(annot_)
            max_element = max_element_i.most_common(1)[0][0]            

        token.append(token_)
        annot.append(max_element)

    # Check if the number of annotations match number of tokens present in the sentence
    assert len(token) == len(annot)
        
    return token, annot