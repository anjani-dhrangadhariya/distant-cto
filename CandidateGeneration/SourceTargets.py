#!/usr/bin/env python
'''
Main python file
'''

import collections
import datetime as dt
import difflib
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from itertools import chain

import matplotlib
import nltk
import numpy as np
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize

matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *

from Align import *
from ExtractData import *
from HeuristicLabelers import *
from Preprocessing import *
from Scoring import *
from MergeAnnot import *

################################################################################
# Set the logger here
################################################################################
LOG_FILE = os.getcwd() + "/logs"
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)

negative_sents = True
MIN_TARGET_LENGTH = 5

file_write_trial = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/intervention_data_preprocessed/' + 'extraction1_pos_posnegtrail.txt'
outdir = '/mnt/nas2/results/Results/systematicReview/Distant_CTO/raw_candidates/'

################################################################################
# Instantiate ElasticSearch
################################################################################
es = Elasticsearch( [{u'host': u'127.0.0.1', u'port': b'9200'}] )

################################################################################
# Get all the documents from the index
################################################################################
# Scan all of the CTO index
results_gen = helpers.scan(
    es,
    query={"query": {"match_all": {}}},
    index='ctofull-index',
    size=1000,
    scroll="60m",
)

match_scores = []

#res = es.search(index="ctofull-index", body={"query": {"match_all": {}}}, size=10)
#print('Total number of records retrieved: ', res['hits']['total']['value'])

with open(f'{outdir}/distantcto_high_conf_09_20dec2021.txt', 'a+') as all_wf:

    for hit in results_gen: # XXX: Entire CTO
    #for n, hit in enumerate( res['hits']['hits'] ): # XXX: Only a part search results from the CTO

        try:

            ################################################################################
            # Initialize annotation dictionaries
            ################################################################################
            aggregated_annotations = collections.defaultdict(dict)
            NCT_id = hit['_source']['FullStudiesResponse']['Expression']
            aggregated_annotations['id'] = NCT_id

            combined_targets = dict()
            combined_sources = dict()
            combined_np_sources = dict()
            combined_bgm_sources = dict()

            fullstudy = hit['_source']['FullStudiesResponse']['FullStudies'][0]['Study']
            protocol_section = fullstudy['ProtocolSection']
            derieved_section = fullstudy['DerivedSection']

            ################################################################################
            # Get and preprocess targets
            ################################################################################
            # target 1: official title
            officialTitleTarget = getOfficialTitle(protocol_section)
            if officialTitleTarget:
                officialTitleTarget = preprocess_targets('officialTitleTarget', officialTitleTarget)
                combined_targets.update(officialTitleTarget)
            
            # target 2: brief title
            briefTitleTarget = getBriefTitle(protocol_section)
            if briefTitleTarget:
                briefTitleTarget = preprocess_targets('briefTitleTarget', briefTitleTarget)
                combined_targets.update(briefTitleTarget)

            # target 3: brief summary
            briefSummaryTarget = getBriefSummary(protocol_section)
            if briefSummaryTarget:
                briefSummaryTarget = preprocess_targets('briefSummaryTarget', briefSummaryTarget)
                combined_targets.update(briefSummaryTarget)

            # target 4: detailed description
            detailedDescriptionTarget = getDetailedDescription(protocol_section)
            if detailedDescriptionTarget:
                detailedDescriptionTarget = preprocess_targets('detailedDescriptionTarget', detailedDescriptionTarget)
                combined_targets.update(detailedDescriptionTarget)

            # target 5: intervention description
            InterventionSource = getInterventionSource(protocol_section)
            for eachInterventionSource in InterventionSource:
                if 'InterventionDescription' in eachInterventionSource:
                    interventionDescription = preprocess_targets('InterventionDescription', eachInterventionSource['InterventionDescription'] )
                    combined_targets.update(interventionDescription)


            ################################################################################
            # Get and preprocess sources
            ################################################################################
            interventionSource = []
            # Source 1: Interventions, Intervention synonyms
            interventions, interventionSyn = getInterventionNames(protocol_section)

            # Source 2: Arms Group Labels will be considered as Intervention names as well
            armGroup = getArmsGroups(protocol_section)

            # Combine the three above-retrieved sources of intervention into a single source
            interventionSource.extend( interventions ); interventionSource.extend( interventionSyn ); interventionSource.extend( armGroup )
            interventionSource = list( set(interventionSource) ) # removes any duplicate entries

            # Process each individual intervention term here. (Abbreviation identification, POS tagging, NP identification, BiGram identification)
            for int_number, eachInterventionTerm in enumerate(interventionSource):
            
                processed_intTerm = preprocess_sources('int_term_', int_number, eachInterventionTerm)
                combined_sources.update(processed_intTerm)
                
                # Fetch Noun Chunks
                possed_np = preprocess_np(processed_intTerm, int_number)
                if possed_np:
                    combined_np_sources.update( possed_np )

                # Fetch Bigrams
                if len(eachInterventionTerm.split(' ')) >= 3:
                    bigrams = getBigrams( eachInterventionTerm )
                    if bigrams:
                        combined_bgm_sources.update( bigrams )

            ################################################################################
            # Add all the targets to the final annotation dictionary
            ################################################################################            
            # LF1: DS labeler
            for key_s, value_s in combined_sources.items():
                
                source_term = value_s['text'].lower()
            
                # Match this source term to each and every target
                for key_t, value_t in combined_targets.items():
                    
                    target_term = value_t['text'].lower()

                    ds_token, ds_annot = align_highconf_target(value_t, source_term)

                    if ds_annot:
                        for annot_type in ['ds', 'ds_np', 'ds_np_fuzz', 'all']:
                            aggregated_annotations = mergeAnnotations(aggregated_annotations, key_t, value_t, annot_type, ds_annot)


                    # LF2: Heuristic ReGeX labeler
                    regex_token, regex_annot = regexMatcher(value_t)

                    if sum(regex_annot) >= 1:
                        for annot_type in ['all']:
                            aggregated_annotations = mergeAnnotations(aggregated_annotations, key_t, value_t, annot_type, regex_annot)


            # LF3: Noun chunk labeler
            for key_s, value_s in combined_np_sources.items():
                
                source_term = value_s['text'].lower()

                # Match this source term to each and every target
                for key_t, value_t in combined_targets.items():
                    
                    target_term = value_t['text'].lower()

                    # LF3 Match Noun Chunks using Distant supervision 
                    np_ds_token, np_ds_annot = align_highconf_target(value_t, source_term)

                    if np_ds_annot:
                        for annot_type in ['ds_np', 'ds_np_fuzz', 'all']:
                            aggregated_annotations = mergeAnnotations(aggregated_annotations, key_t, value_t, annot_type, np_ds_annot)


            # LF4: Fuzzy Bigram labeler TODO: Convert funnzy bigram matcher to fuzzy 8-gram matcher
            for key_s, value_s in combined_bgm_sources.items():
                
                source_term = value_s['text'].lower()

                # Match this source term to each and every target
                for key_t, value_t in combined_targets.items():
                    
                    target_term = value_t['text'].lower()

                    bgm_ds_token, bgm_ds_annot = align_highconf_target(value_t, source_term)

                    if bgm_ds_annot:
                        for annot_type in ['ds_np_fuzz', 'all']:
                            aggregated_annotations = mergeAnnotations(aggregated_annotations, key_t, value_t, annot_type, bgm_ds_annot)

            if aggregated_annotations:
                logNCTID = 'Writing ID: ' + NCT_id
                logging.info(logNCTID)
                all_wf.write('\n')
                json_str = json.dumps(aggregated_annotations)
                all_wf.write(json_str)

        except Exception as ex:

            template = "An exception of type {0} occurred. Arguments:{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print( NCT_id , ' : ', message )

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

            print(traceback.format_exc())

            logNCTID = 'Caused exception at the NCT ID: ' + NCT_id
            logging.info(logNCTID)