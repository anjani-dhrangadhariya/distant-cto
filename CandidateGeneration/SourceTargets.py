#!/usr/bin/env python
'''
Main python file
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
import traceback

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *

from Preprocessing import *
from ExtractData import *
from Scoring import *
from Align import *
from HeuristicLabelers.HeuristicLabel import *

################################################################################
# Set the logger here
################################################################################
LOG_FILE = os.getcwd() + "/logs"
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)

LOG_FILE = LOG_FILE + "/" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d %H_%M_%S') + ".log"
logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
fileHandler = logging.FileHandler("{0}".format(LOG_FILE))
rootLogger = logging.getLogger()
rootLogger.addHandler(fileHandler)
rootLogger.setLevel(logging.INFO)

negative_sents = True
MIN_TARGET_LENGTH = 5

################################################################################
# Local functions
################################################################################
# Partitiion a list
# def partition(list_in, n):
#     # random.shuffle(list_in)
#     return [list_in[i::n] for i in range(n)]

def partition(lst, n):
    """Return successive n-sized chunks from list (lst)."""
    chunks = []
    for i in range(0, len(lst), n):
        chunks.append( lst[i:i + n]  )
    return chunks

# aggregate the annotations here
def aggregateLongTarget_annot(agrregateannot_briefSummary):
    """Aggregate annotations from multiple intervention sources for each target."""
    briefsummary_aggdict = dict()
    for eachAggAnnot in agrregateannot_briefSummary:
        sentenceKey = list(eachAggAnnot.keys())
        for eachsentenceKey in sentenceKey:
            if eachsentenceKey not in briefsummary_aggdict:
                briefsummary_aggdict[eachsentenceKey] = eachAggAnnot[eachsentenceKey]
            elif eachsentenceKey in briefsummary_aggdict:
                annotUpdater = eachAggAnnot[eachsentenceKey]
                for count, eachItem  in enumerate(annotUpdater[1]):
                    if eachItem == 1:
                        briefsummary_aggdict[eachsentenceKey][1][count] = 1
    return briefsummary_aggdict

def pos_neg_trail(aggregated_dictionary):
    """Generate and return +- trailing annotations."""
    values_ = []
    for key, value in aggregated_dictionary.items():
        values_.extend( value[1] )

    mergedChunks_dictionary = dict()
    
    if 1 in values_:
        # Sort the dictionary
        aggregated_dictionary_sorted = collections.OrderedDict(sorted(aggregated_dictionary.items()))

        # Partition into chunks
        chunks = partition( list(aggregated_dictionary_sorted.keys()), 4)

        # Merge each chunk into a single key-value pair
        for eachChunk in chunks:
            keyChunks = []
            valueChunk_sent = []
            valueChunk_lab = []
            valueChunk_pos = []
            for eachChunk_i in eachChunk:
                keyChunks.append( eachChunk_i )
                valueChunk_sent.extend( aggregated_dictionary_sorted[eachChunk_i][0] )
                valueChunk_lab.extend( aggregated_dictionary_sorted[eachChunk_i][1] )
                valueChunk_pos.extend( aggregated_dictionary_sorted[eachChunk_i][2] )

            assert len(valueChunk_sent) == len(valueChunk_lab) == len(valueChunk_pos)

            mergedKey = str('_'.join(keyChunks))
            mergedChunks_dictionary[mergedKey] = [valueChunk_sent, valueChunk_lab, valueChunk_pos]

    if bool(mergedChunks_dictionary) == True:
        return mergedChunks_dictionary

file_write_trial = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/intervention_data_preprocessed/' + 'extraction1_pos_posnegtrail.txt'

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
intervention_types = []

res = es.search(index="ctofull-index", body={"query": {"match_all": {}}}, size=10)
print('Total number of records retrieved: ', res['hits']['total']['value'])
# for hit in results_gen: # XXX: Entire CTO
for n, hit in enumerate( res['hits']['hits'] ): # XXX: Only a part search results from the CTO

    write_hit = collections.defaultdict(dict) # final dictionary to write to the file...

    fullstudy = hit['_source']['FullStudiesResponse']['FullStudies'][0]['Study']
    NCT_id = hit['_source']['FullStudiesResponse']['Expression']
    write_hit['id'] = NCT_id

    try:

        protocol_section = fullstudy['ProtocolSection']
        derieved_section = fullstudy['DerivedSection']

        ################################################################################
        # Get and preprocess targets (XXX targets are singular unstructured strings)
        ################################################################################
        # target 1: official title
        officialTitleTarget = getOfficialTitle(protocol_section)
        if officialTitleTarget:
            officialTitleTarget = preprocess_targets(officialTitleTarget)
        
        # target 2: brief title
        briefTitleTarget = getBriefTitle(protocol_section)
        if briefTitleTarget:
            briefTitleTarget = preprocess_targets(briefTitleTarget)

        # target 3: brief summary
        briefSummaryTarget = getBriefSummary(protocol_section)
        if briefSummaryTarget:
            briefSummaryTarget = preprocess_targets(briefSummaryTarget)

        # target 4: detailed description
        detailedDescriptionTarget = getDetailedDescription(protocol_section)
        if detailedDescriptionTarget:
            detailedDescriptionTarget = preprocess_targets(detailedDescriptionTarget)


        ################################################################################
        # Get and preprocess sources (XXX sources are plural/multiple structured terms)
        ################################################################################
        """interventionSource generator contains multiple interventions used in the study.
        Each interventionSource generator instance contains meta-information about the intervention used."""
        # Source 1: Interventions
        interventionSource = getInterventionNames(protocol_section)

        # Source 2: Arms Groups
        armGroup = getArmsGroups(protocol_section)

        # Aggregate annotations for each target text
        agrregateannot_officialTitleTarget = []
        agrregateannot_briefTitleTarget = []
        agrregateannot_interventionDescription = []
        agrregateannot_briefSummary = []
        agrregateannot_detailedDescription = [] 

        intervention_counter = 0
        # XXX Each individual intervention term is iterated here
        for eachInterventionSource in interventionSource:
          
            write_intervention = dict() # Write all the intervention annotations for any particular hit in this dictionary
          
            intervention_counter = intervention_counter + 1
            write_intervention['intervention_number'] = intervention_counter

            # Intervention type 
            interventionType = eachInterventionSource['InterventionType']
            write_intervention['intervention_type'] = interventionType
            intervention_types.append(interventionType)

            # Source 1.1: Intervention Name
            if 'InterventionName' in eachInterventionSource:
                interventionName = eachInterventionSource['InterventionName']
                interventionName = interventionName.replace('-', '')
                write_intervention['intervention_name'] = interventionName

                # Get POS-tag for each Intervention name
                possed_interventionName = getPOStags( interventionName )
            else: 
                interventionName = None
            
            # Source 1.2: Intervention Other name
            if 'InterventionOtherName' in eachInterventionSource:
                interventionOtherNames = eachInterventionSource['InterventionOtherName']
            else:
                interventionOtherNames = None

            # target 5: intervention description
            if 'interventiondescription' in eachInterventionSource:
                interventionDescription = preprocess_targets( eachInterventionSource['interventiondescription'] )
            else:
                interventionDescription = None


            if possed_interventionName is not None:

                # Source
                interventionName = possed_interventionName['text'].lower()

                ####################################annotations​##################################################
                # Candidate Generation 1: Only Intervention names
                ######################################################################################
                ######################################################################################
                # Match the source intervention to the official title
                ######################################################################################
                if officialTitleTarget is not None and len( officialTitleTarget['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                    officialTitleTarget_token, officialTitleTarget_annot = align_highconf_shorttarget(officialTitleTarget['text'], interventionName)

                    regex_token, regex_annot = regexMatcher(officialTitleTarget)

                    if officialTitleTarget_annot:
                        write_hit['aggregate_annot']['official_title'] = officialTitleTarget_token
                        write_hit['aggregate_annot']['official_title_pos'] = [eachTuple[1]  for eachTuple in nltk.pos_tag_sents([officialTitleTarget_token])[0]]
                        write_intervention['official_title'] = officialTitleTarget_token
                        write_intervention['official_title_annot'] = officialTitleTarget_annot
                        if not agrregateannot_officialTitleTarget:
                            agrregateannot_officialTitleTarget.extend(officialTitleTarget_annot)
                        elif agrregateannot_officialTitleTarget:
                            for count, eachItem in enumerate(officialTitleTarget_annot):
                                if eachItem == 1:
                                    agrregateannot_officialTitleTarget[count] = 1

                ######################################################################################
                # Match the source intervention to the brief title
                ######################################################################################
                
                if briefTitleTarget is not None and len( briefTitleTarget['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                    briefTitleTarget_token, briefTitleTarget_annot = align_highconf_shorttarget(briefTitleTarget['text'], interventionName)

                    if briefTitleTarget_annot:
                        write_hit['aggregate_annot']['brief_title'] = briefTitleTarget_token
                        write_hit['aggregate_annot']['brief_title_pos'] = [eachTuple[1]  for eachTuple in nltk.pos_tag_sents([briefTitleTarget_token])[0]]
                        write_intervention['brief_title'] = briefTitleTarget_token
                        write_intervention['brief_title_annot'] = briefTitleTarget_annot
                        if not agrregateannot_briefTitleTarget:
                            agrregateannot_briefTitleTarget.extend(briefTitleTarget_annot)
                        elif agrregateannot_briefTitleTarget:
                            for count, eachItem in enumerate(briefTitleTarget_annot):
                                if eachItem == 1:
                                    agrregateannot_briefTitleTarget[count] = 1

                ######################################################################################
                # Match the source intervention to the intervention description
                ######################################################################################  
               
                if interventionDescription is not None and len( interventionDescription['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                    interventionDescription_annot_ = align_highconf_longtarget(interventionDescription['text'], interventionName)

                    interventionDescription_annot = dict()
                    for key, value in interventionDescription_annot_.items():
                        if len(value[0]) >= 1:
                            interventionDescription_annot[key] = value

                    if interventionDescription_annot:
                        if 'interventionDescription_annot' not in write_intervention:
                            write_intervention['interventionDescription_annot'] = [interventionDescription_annot]
                            agrregateannot_interventionDescription.append( interventionDescription_annot )
                        elif 'interventionDescription_annot' in write_intervention:
                            write_intervention['interventionDescription_annot'].append(interventionDescription_annot)
                            agrregateannot_interventionDescription.append( interventionDescription_annot )

                ######################################################################################
                # Match the source intervention to the brief summary
                ######################################################################################
                if briefSummaryTarget is not None and len( briefSummaryTarget['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                    briefSummaryTarget_annot = align_highconf_longtarget_negSent(briefSummaryTarget['text'], interventionName)
                    if briefSummaryTarget_annot:
                        if 'brief_summary_annot' not in write_intervention:
                            write_intervention['brief_summary_annot'] = [briefSummaryTarget_annot]
                            agrregateannot_briefSummary.append( briefSummaryTarget_annot )
                        elif 'brief_summary_annot' in write_intervention:
                            write_intervention['brief_summary_annot'].append(briefSummaryTarget_annot)
                            agrregateannot_briefSummary.append( briefSummaryTarget_annot )

                ######################################################################################
                # Match the source intervention to the detailed description
                ######################################################################################
                if detailedDescriptionTarget is not None and len( detailedDescriptionTarget['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                    detailedDescriptionTarget_annot = align_highconf_longtarget_negSent(detailedDescriptionTarget['text'], interventionName)
                    if detailedDescriptionTarget_annot:
                        write_intervention['detailed_description_annot'] = [detailedDescriptionTarget_annot]
                        agrregateannot_detailedDescription.append( detailedDescriptionTarget_annot )
                    elif 'detailed_description_annot' in write_intervention:
                        write_intervention['detailed_description_annot'].append(detailedDescriptionTarget_annot)
                        agrregateannot_detailedDescription.append( detailedDescriptionTarget_annot )

                # The main intervention term is tackled. Now tackle the intervention synonyms...
                #####################################################################################
                #  Candidate Generation 2: Intervention other names
                #####################################################################################

                if interventionOtherNames:

                    interventionSynonyms = interventionOtherNames['interventionothername']

                    # add a sub-dict to the "write_intervention" dictionary
                    write_intervention_syn = dict()

                    for i, eachInterventionOtherName in enumerate(interventionSynonyms):

                        write_intervention_syn['synonym_name'] = eachInterventionOtherName

                        ######################################################################################
                        # Match the source intervention to the official title
                        #######################################################################################
                        if officialTitleTarget is not None and len( officialTitleTarget['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                            officialTitleTarget_syntoken, officialTitleTarget_synannot = align_highconf_shorttarget(officialTitleTarget['text'], eachInterventionOtherName)
                            if officialTitleTarget_synannot:
                                write_hit['aggregate_annot']['official_title'] = officialTitleTarget_syntoken
                                write_hit['aggregate_annot']['official_title_pos'] = [eachTuple[1]  for eachTuple in nltk.pos_tag_sents([officialTitleTarget_syntoken])[0]]
                                write_intervention_syn['official_title'] = officialTitleTarget_syntoken
                                write_intervention_syn['official_title_annot'] = officialTitleTarget_synannot
                                if not agrregateannot_officialTitleTarget:
                                    agrregateannot_officialTitleTarget.extend(officialTitleTarget_synannot)
                                elif agrregateannot_officialTitleTarget:
                                    for count, eachItem in enumerate(officialTitleTarget_synannot):
                                        if eachItem == 1:
                                            agrregateannot_officialTitleTarget[count] = 1

                        ######################################################################################
                        # Match the source intervention to the brief title
                        ######################################################################################
                        if briefTitleTarget is not None and len( officialTitleTarget['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                            briefTitleTarget_syntoken, briefTitleTarget_synannot = align_highconf_shorttarget(briefTitleTarget['text'], eachInterventionOtherName)
                            if briefTitleTarget_synannot:
                                write_hit['aggregate_annot']['brief_title'] = briefTitleTarget_syntoken
                                write_hit['aggregate_annot']['brief_title_pos'] = [eachTuple[1]  for eachTuple in nltk.pos_tag_sents([briefTitleTarget_syntoken])[0]]
                                write_intervention_syn['brief_title'] = briefTitleTarget_syntoken
                                write_intervention_syn['brief_title_annot'] = briefTitleTarget_synannot
                                if not agrregateannot_briefTitleTarget:
                                    agrregateannot_briefTitleTarget.extend(briefTitleTarget_synannot)
                                elif agrregateannot_briefTitleTarget:
                                    for count, eachItem in enumerate(briefTitleTarget_synannot):
                                        if eachItem == 1:
                                            agrregateannot_briefTitleTarget[count] = 1

                        ######################################################################################
                        # Match the source intervention to the intervention description
                        ######################################################################################
                        if interventionDescription is not None and len( interventionDescription['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                            interventionDescription_synannot_ = align_highconf_longtarget(interventionDescription['text'], eachInterventionOtherName) 
                            interventionDescription_synannot = dict()
                            for key, value in interventionDescription_synannot_.items():
                                if len(value[0]) >= 1:
                                    interventionDescription_synannot[key] = value

                            if interventionDescription_synannot:
                                if 'interventionDescription_annot' not in write_intervention_syn:
                                    write_intervention_syn['interventionDescription_annot'] = [interventionDescription_synannot]
                                    agrregateannot_interventionDescription.append( interventionDescription_synannot )
                                elif 'interventionDescription_annot' in write_intervention_syn:
                                    write_intervention_syn['interventionDescription_annot'].append(interventionDescription_synannot)
                                    agrregateannot_interventionDescription.append( interventionDescription_synannot )

                        ######################################################################################
                        # Match the source intervention to the brief summary
                        ######################################################################################
                        if briefSummaryTarget is not None and len( briefSummaryTarget['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                            briefSummaryTarget_synannot  = align_highconf_longtarget_negSent(briefSummaryTarget['text'], eachInterventionOtherName)
                            if briefSummaryTarget_synannot:
                                if 'brief_summary_annot' not in write_intervention_syn:
                                    write_intervention_syn['brief_summary_annot'] = [briefSummaryTarget_synannot]
                                    agrregateannot_briefSummary.append( briefSummaryTarget_synannot )
                                elif 'brief_summary_annot' in write_intervention_syn:
                                    write_intervention_syn['brief_summary_annot'].append(briefSummaryTarget_synannot)
                                    agrregateannot_briefSummary.append( briefSummaryTarget_synannot )
                                    # tempAnnot = write_intervention['brief_summary_annot']
                                    # agrregateannot_briefSummary.append(tempAnnot)

                        ######################################################################################
                        # Match the source intervention to the detailed description
                        ######################################################################################
                        if detailedDescriptionTarget is not None and len( detailedDescriptionTarget['text'].split(' ') ) >= MIN_TARGET_LENGTH:
                            detailedDescriptionTarget_synannot  = align_highconf_longtarget_negSent(detailedDescriptionTarget['text'], eachInterventionOtherName)
                            if detailedDescriptionTarget_synannot:
                                if 'detailed_description_annot' in write_intervention_syn:
                                    write_intervention_syn['detailed_description_annot'] = [detailedDescriptionTarget_synannot]
                                    agrregateannot_detailedDescription.append( detailedDescriptionTarget_synannot )
                                elif 'detailed_description_annot' in write_intervention_syn:
                                    write_intervention_syn['detailed_description_annot'].append(detailedDescriptionTarget_synannot)
                                    agrregateannot_detailedDescription.append( detailedDescriptionTarget_synannot )
                                    # tempAnnot = write_intervention['detailed_description_annot']
                                    # agrregateannot_detailedDescription.append(tempAnnot)
                        

                        # Add to the "write_intervention" here
                        subInterventionCounter = 'syn_' + str(i)
                        write_intervention[subInterventionCounter] = write_intervention_syn

                # # Write the intervention section to the hit dictionary
                write_hit['extraction1'][intervention_counter] = write_intervention

            if agrregateannot_officialTitleTarget:
                write_hit['aggregate_annot']['official_title_annot'] = agrregateannot_officialTitleTarget

            if agrregateannot_briefTitleTarget:
                write_hit['aggregate_annot']['brief_title_annot'] = agrregateannot_briefTitleTarget

            if agrregateannot_interventionDescription:
                interventionDescription_aggdict = aggregateLongTarget_annot(agrregateannot_interventionDescription)
                if interventionDescription_aggdict:
                    write_hit['aggregate_annot']['intervention_description_annot'] = interventionDescription_aggdict

            if agrregateannot_briefSummary:
                briefsummary_aggdict = aggregateLongTarget_annot(agrregateannot_briefSummary)
                
                # Mix/merge the aggregated dictionary for +- training
                if negative_sents == True:
                    briefsummary_aggdict = pos_neg_trail( briefsummary_aggdict )

                if briefsummary_aggdict:
                    write_hit['aggregate_annot']['brief_summary_annot'] = briefsummary_aggdict

            if agrregateannot_detailedDescription:
                detailedDescription_aggdict = aggregateLongTarget_annot(agrregateannot_detailedDescription)

                # Mix/merge the aggregated dictionary for +- training
                if negative_sents == True:
                    detailedDescription_aggdict = pos_neg_trail( detailedDescription_aggdict )

                if detailedDescription_aggdict:
                    write_hit['aggregate_annot']['detailed_description_annot'] = detailedDescription_aggdict

        logNCTID = 'Writing ID: ' + NCT_id
        logging.info(logNCTID)
        # with open(file_write_trial, 'a+') as wf:
        #     wf.write('\n')
        #     json_str = json.dumps(write_hit)
        #     wf.write(json_str)

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