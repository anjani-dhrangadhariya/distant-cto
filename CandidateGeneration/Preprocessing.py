#!/usr/bin/env python
'''
This module contains text preprocessing functions used to preprocess and normalize text (Source and Target) before they could be aligned or matched.
'''

################################################################################
# Library imports
################################################################################
import enum
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from spacy.lang.en import English
from scispacy.abbreviation import AbbreviationDetector
stop_words = set(stopwords.words('english'))

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("sentencizer")
nlp.add_pipe("abbreviation_detector")
stopwords = nlp.Defaults.stop_words
additional_stopwords = ['of']
stopwords.update(additional_stopwords)

################################################################################
# Preprocessing functions
################################################################################
def fetch_tokens(s):
    doc = nlp(s)
    return [ token.text for token in doc ]


'''
Description:
    The function extracts acronyms using Scispacy abbreviation detector

Args:
    string (str): free-text string

Returns:
    List (list): tuple with detected abbreviation and the long form of the abbreviation
'''
def fetchAcronyms(value):

    doc = nlp(value)

    altered_tok = [tok.text for tok in doc]

    abbreviations = []

    for abrv in doc._.abbreviations:
        altered_tok[abrv.start] = str(abrv._.long_form)

        # print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form} \t  {value}")
        # print( " ".join(altered_tok) )

        abbreviations.append( str(abrv) )
        abbreviations.append( str(abrv._.long_form) )
        # original = str(abrv._.long_form) + ' (' + str(abrv) + ')'
        # abbreviations.append( original  )

    if abbreviations:
        return abbreviations

'''
Description:
    The function fetches POS tags for the input string
Args:
    String: free-text string
Returns:
    Dictionary (dict): returns a dictionary containing free-text string with its tokenized string, token lemma, POS tags for the tokens, finer POS tags for the token
'''
def getPOStags(value):

    doc = nlp(value)

    pos_dict = dict()

    tokens = [ token.text for token in doc ]
    lemma = [ token.lemma_ for token in doc ]
    pos = [ token.pos_ for token in doc ]
    pos_fine = [ token.tag_ for token in doc ]
    depth = [ token.dep_ for token in doc ]
    shape = [ token.shape_ for token in doc ]
    is_alpha = [ token.is_alpha for token in doc ]
    is_stop = [ token.is_stop for token in doc ]
    char_offsets = [token.idx for token in doc]
    noun_chunks = [str(chunk) for chunk in doc.noun_chunks]

    assert len( char_offsets ) == len( tokens )

    pos_dict['text'] = value
    pos_dict['tokens'] = tokens
    pos_dict['lemma'] = lemma
    pos_dict['pos'] = pos
    pos_dict['pos_fine'] = pos_fine
    pos_dict['idx'] = char_offsets
    pos_dict['noun_chunks'] = noun_chunks

    return pos_dict

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

def removePunct(s):
    return s.translate(str.maketrans(' ', ' ', string.punctuation))


def clean_unicide(s):
    s_encode = s.encode("ascii", "ignore")
    return s_encode.decode()

def preprocess_targets(target_nomen, s):

    targets = dict()
    
    modified_s = removeHyphenString(s) 
    modified_s = removeSpaceTrailsString(modified_s)
    modified_s = clean_unicide(modified_s)

    doc = nlp(modified_s)

    for i, sent in enumerate(list(doc.sents)):
        key = target_nomen + '_' + str(i)
        targets[key] = getPOStags( str(sent) )

    return targets


def preprocess_sources(source_nomen, counter, s):

    sources = dict()

    modified_s = clean_unicide(s)

    detected_abb = fetchAcronyms(modified_s)

    if detected_abb:
        for i, abb in enumerate(detected_abb):
            key = source_nomen + str(counter) + str(i)
            sources[key] = getPOStags( str(abb) )
    else:
        key = source_nomen + str(counter)
        sources[key] = getPOStags( str(modified_s) )

    return sources

def preprocess_np(int_with_np, counter_key):

    possed_interventionNounChunk = dict()

    for k,v in int_with_np.items():
        if 'noun_chunks' in v and len(v['noun_chunks']) >= 2:
            for chunk_number, chunk in enumerate(v['noun_chunks']):
                key = str(counter_key) + '_' + str(chunk_number)
                chunk_possed = preprocess_sources('np_', key, chunk)
                possed_interventionNounChunk.update( chunk_possed )
    
    return possed_interventionNounChunk


def ngrams(s):

    bigrams = []

    tokens = fetch_tokens(s)
    removed_stopwords = ' '.join([x for x in tokens if x not in stop_words ])
    bigrm = nltk.bigrams(removed_stopwords.split(' '))

    for bgm in bigrm:
        bigrams.append( ' '.join(bgm) )

    return bigrams


def getBigrams(intervention_term):

    # Preprocess intervention terms for removal of stopwords
    intervention_term = [ t for t in intervention_term.split() if t not in stopwords ]
    intervention_term = ' '.join( intervention_term )

    bigram_dict = dict()
    bigrams = []
    
    detected_abb = fetchAcronyms(intervention_term)

    if detected_abb:
        for i, abb in enumerate(detected_abb):
            if len(abb.split(' ')) >= 3: # If there are actually more words than 2
                punct_stripped = removePunct(abb)
                modified_term = removeSpaceTrailsString(punct_stripped)
                bigrams.extend( ngrams(modified_term) )

    else:
        punct_stripped = removePunct(intervention_term)
        modified_term = removeSpaceTrailsString(punct_stripped)
        bigrams.extend( ngrams(modified_term) ) 

    if bigrams:
        for i, bgm in enumerate(bigrams):
            possed_bgm = getPOStags(bgm)
            key = 'bigram_'+ str(i)
            bigram_dict[key] = possed_bgm

    return bigram_dict