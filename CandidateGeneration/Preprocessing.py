#!/usr/bin/env python
'''
This module contains text preprocessing functions used to preprocess and normalize text (Source and Target) before they could be aligned or matched.
'''
__author__ = "Anjani Dhrangadhariya"
__maintainer__ = "Anjani Dhrangadhariya"
__email__ = "anjani.k.dhrangadhariya@gmail.com"
__status__ = "Prototype/Research"

################################################################################
# Library imports
################################################################################
import re

################################################################################
# Preprocessing functions
################################################################################
# Lower case source and targets
def lowercaseString(s):
    return s.lower()

def lowercaseDict(s):
        #return dict((k, v.lower()) for k,v in s.items())
        return dict(eval(repr(s).lower()))

# Remove punctuations
def removeHyphenString(s):
    return s.replace('-', ' ')

def removeSpaceTrailsString(s):
    return " ".join(s.split())


def clean_unicide(s):
    s_encode = s.encode("ascii", "ignore")
    return s_encode.decode()

# Normalize symbols like + (plus)

def preprocess_targets(s):

    if isinstance(s, str):
        modified_s = lowercaseString(s)

    modified_s = removeSpaceTrailsString(modified_s)
    modified_s = removeHyphenString(modified_s) 
    modified_s = clean_unicide(modified_s)    

    return modified_s