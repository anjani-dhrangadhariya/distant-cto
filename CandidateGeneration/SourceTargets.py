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
from Align import *

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

################################################################################
# Local functions
################################################################################
# aggregate the annotations here
def aggregateLongTarget_annot(agrregateannot_briefSummary):
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

file_write_trial = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/intervention_data_preprocessed/' + 'extraction1.txt'

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
# sanity_counter = 0

res = es.search(index="ctofull-index", body={"query": {"match_all": {}}}, size=5000)
print('Total number of records retrieved: ', res['hits']['total']['value'])
# for hit in results_gen: # XXX: Entire CTO
for hit in res['hits']['hits']: # XXX: Only a part search results from the CTO

    write_hit = collections.defaultdict(dict) # final dictionary to write to the file...
    aggregate_extractions = dict() # collect or aggregate all the intervention annotations for a record region-wise here.

    fullstudy = hit['_source']['FullStudiesResponse']['FullStudies'][0]['Study']
    NCT_id = hit['_source']['FullStudiesResponse']['Expression']
    write_hit['id'] = NCT_id

    try:
        # if 'NCT00105235' in NCT_id:
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

            # Lower case the intervention source dictionary
            eachInterventionSource = lowercaseDict(eachInterventionSource)
            
            intervention_counter = intervention_counter + 1
            write_intervention['intervention_number'] = intervention_counter
            # print('________________________________________________________________________________   ', intervention_counter)

            # Intervention type 
            interventionType = eachInterventionSource['interventiontype']
            write_intervention['intervention_type'] = interventionType
            intervention_types.append(interventionType)
            
            # Source 1.1: Intervention Name
            if 'interventionname' in eachInterventionSource:
                interventionName = eachInterventionSource['interventionname']
                interventionName = interventionName.replace('-', '')
                write_intervention['intervention_name'] = interventionName
            else: 
                interventionName = None
            # print(interventionName)
            
            # Source 1.2: Intervention Other name
            if 'interventionothernamelist' in eachInterventionSource:
                interventionOtherNames = eachInterventionSource['interventionothernamelist']
            else:
                interventionOtherNames = None

            # target 5: intervention description
            if 'interventiondescription' in eachInterventionSource:
                interventionDescription = preprocess_targets( eachInterventionSource['interventiondescription'] )
            else:
                interventionDescription = None

            if interventionName is not None:

                ####################################​##################################################
                # Candidate Generation 1: Only Intervention names
                ######################################################################################
                ######################################################################################
                # Match the source intervention to the official title
                ######################################################################################

                # print(officialTitleTarget)
                officialTitleTarget_token, officialTitleTarget_annot = align_highconf_shorttarget(officialTitleTarget, interventionName)
                # if officialTitleTarget_annot:
                #     write_hit['aggregate_annot']['official_title'] = officialTitleTarget_token
                #     write_intervention['official_title'] = officialTitleTarget_token
                #     write_intervention['official_title_annot'] = officialTitleTarget_annot
                    # if not agrregateannot_officialTitleTarget:
                    #     agrregateannot_officialTitleTarget.extend(officialTitleTarget_annot)
                    # elif agrregateannot_officialTitleTarget:
                    #     for count, eachItem in enumerate(officialTitleTarget_annot):
                    #         if eachItem == 1:
                    #             agrregateannot_officialTitleTarget[count] = 1

                ######################################################################################
                # Match the source intervention to the brief title
                ######################################################################################
                briefTitleTarget_token, briefTitleTarget_annot = align_highconf_shorttarget(briefTitleTarget, interventionName)
                # if briefTitleTarget_annot:
                #     write_hit['aggregate_annot']['brief_title'] = briefTitleTarget_token
                #     write_intervention['brief_title'] = briefTitleTarget_token
                #     write_intervention['brief_title_annot'] = briefTitleTarget_annot
                    # if not agrregateannot_briefTitleTarget:
                    #     agrregateannot_briefTitleTarget.extend(briefTitleTarget_annot)
                    # elif agrregateannot_briefTitleTarget:
                    #     for count, eachItem in enumerate(briefTitleTarget_annot):
                    #         if eachItem == 1:
                    #             agrregateannot_briefTitleTarget[count] = 1
            
                ######################################################################################
                # Match the source intervention to the intervention description
                ###################################################token, annot###################################  
                
                interventionDescription_token, interventionDescription_annot = align_highconf_shorttarget(interventionDescription, interventionName)                       
                # #print('Intervention description target: ', interventionDescription_annot)
                # if interventionDescription_annot:
                #     write_intervention['intervention_description'] = interventionDescription_token
                #     write_intervention['intervention_description_annot'] = interventionDescription_annot
                #     agrregateannot_interventionDescription.append( [interventionDescription_token, interventionDescription_annot] )

                ######################################################################################
                # Match the source intervention to the brief summary
                ######################################################################################
                if briefSummaryTarget:
                    briefSummaryTarget_annot = align_highconf_longtarget(briefSummaryTarget, interventionName)
                #     # print(type(briefSummaryTarget_annot))
                #     # print(briefSummaryTarget_annot)
                #     if briefSummaryTarget_annot:
                #         if 'brief_summary_annot' not in write_intervention:
                #             write_intervention['brief_summary_annot'] = [briefSummaryTarget_annot]
                #             agrregateannot_briefSummary.append( briefSummaryTarget_annot )
                #         elif 'brief_summary_annot' in write_intervention:
                #             write_intervention['brief_summary_annot'].append(briefSummaryTarget_annot)
                #             agrregateannot_briefSummary.append( briefSummaryTarget_annot )

                ######################################################################################
                # Match the source intervention to the detailed description
                ######################################################################################
                if detailedDescriptionTarget:
                    detailedDescriptionTarget_annot = align_highconf_longtarget(detailedDescriptionTarget, interventionName)
                #     if detailedDescriptionTarget_annot:
                #         write_intervention['detailed_description_annot'] = [detailedDescriptionTarget_annot]
                #         agrregateannot_detailedDescription.append( detailedDescriptionTarget_annot )
                #     elif 'detailed_description_annot' in write_intervention:
                #         write_intervention['detailed_description_annot'].append(detailedDescriptionTarget_annot)
                #         agrregateannot_detailedDescription.append( detailedDescriptionTarget_annot )

                # The main intervention term is tackled. Now tackle the intervention synonyms...
                #####################################################################################
                #  Candidate Generation 2: Intervention other names
                #####################################################################################

                # if interventionOtherNames:

                #     interventionSynonyms = interventionOtherNames['interventionothername']

                #     # add a sub-dict to the "write_intervention" dictionary
                #     write_intervention_syn = dict()

                #     for i, eachInterventionOtherName in enumerate(interventionSynonyms):

                #         write_intervention_syn['synonym_name'] = eachInterventionOtherName

                #         ######################################################################################
                #         # Match the source intervention to the official title
                #         #######################################################################################
                #         officialTitleTarget_syntoken, officialTitleTarget_synannot = align_highconf_shorttarget(officialTitleTarget, eachInterventionOtherName)
                #         if officialTitleTarget_synannot:
                #             write_hit['aggregate_annot']['official_title'] = officialTitleTarget_syntoken
                #             write_intervention_syn['official_title'] = officialTitleTarget_syntoken
                #             write_intervention_syn['official_title_annot'] = officialTitleTarget_synannot
                #             if not agrregateannot_officialTitleTarget:
                #                 agrregateannot_officialTitleTarget.extend(officialTitleTarget_synannot)
                #             elif agrregateannot_officialTitleTarget:
                #                 for count, eachItem in enumerate(officialTitleTarget_synannot):
                #                     if eachItem == 1:
                #                         agrregateannot_officialTitleTarget[count] = 1

                #         ######################################################################################
                #         # Match the source intervention to the brief title
                #         ######################################################################################
                        
                #         briefTitleTarget_syntoken, briefTitleTarget_synannot = align_highconf_shorttarget(briefTitleTarget, eachInterventionOtherName)
                #         if briefTitleTarget_synannot:
                #             write_hit['aggregate_annot']['brief_title'] = briefTitleTarget_syntoken
                #             write_intervention_syn['brief_title'] = briefTitleTarget_syntoken
                #             write_intervention_syn['brief_title_annot'] = briefTitleTarget_synannot
                #             if not agrregateannot_briefTitleTarget:
                #                 agrregateannot_briefTitleTarget.extend(briefTitleTarget_synannot)
                #             elif agrregateannot_briefTitleTarget:
                #                 for count, eachItem in enumerate(briefTitleTarget_synannot):
                #                     if eachItem == 1:
                #                         agrregateannot_briefTitleTarget[count] = 1

                #         ######################################################################################
                #         # Match the source intervention to the intervention description
                #         ######################################################################################  
                        
                #         interventionDescription_syntoken, interventionDescription_synannot = align_highconf_shorttarget(interventionDescription, eachInterventionOtherName) 
                #         if interventionDescription_synannot:
                #             write_intervention_syn['intervention_description'] = interventionDescription_syntoken
                #             write_intervention_syn['intervention_description_annot'] = interventionDescription_synannot
                #             agrregateannot_interventionDescription.append( [interventionDescription_syntoken, interventionDescription_synannot] )

                #         ######################################################################################
                #         # Match the source intervention to the brief summary
                #         ######################################################################################
                #         if briefSummaryTarget:
                #             briefSummaryTarget_synannot  = align_highconf_longtarget(briefSummaryTarget, eachInterventionOtherName)
                #             if briefSummaryTarget_synannot:
                #                 if 'brief_summary_annot' not in write_intervention_syn:
                #                     write_intervention_syn['brief_summary_annot'] = [briefSummaryTarget_synannot]
                #                     agrregateannot_briefSummary.append( briefSummaryTarget_synannot )
                #                 elif 'brief_summary_annot' in write_intervention_syn:
                #                     write_intervention_syn['brief_summary_annot'].append(briefSummaryTarget_synannot)
                #                     agrregateannot_briefSummary.append( briefSummaryTarget_synannot )
                #                     # tempAnnot = write_intervention['brief_summary_annot']
                #                     # agrregateannot_briefSummary.append(tempAnnot)

                #         ######################################################################################
                #         # Match the source intervention to the detailed description
                #         ######################################################################################
                #         if detailedDescriptionTarget:
                #             detailedDescriptionTarget_synannot  = align_highconf_longtarget(detailedDescriptionTarget, eachInterventionOtherName)
                #             if detailedDescriptionTarget_synannot:
                #                 if 'detailed_description_annot' in write_intervention_syn:
                #                     write_intervention_syn['detailed_description_annot'] = [detailedDescriptionTarget_synannot]
                #                     agrregateannot_detailedDescription.append( detailedDescriptionTarget_synannot )
                #                 elif 'detailed_description_annot' in write_intervention_syn:
                #                     write_intervention_syn['detailed_description_annot'].append(detailedDescriptionTarget_synannot)
                #                     agrregateannot_detailedDescription.append( detailedDescriptionTarget_synannot )
                #                     # tempAnnot = write_intervention['detailed_description_annot']
                #                     # agrregateannot_detailedDescription.append(tempAnnot)
                        

                #         # Add to the "write_intervention" here
                #         subInterventionCounter = 'syn_' + str(i)
                #         write_intervention[subInterventionCounter] = write_intervention_syn

                # # Write the intervention section to the hit dictionary
                # write_hit['extraction1'][intervention_counter] = write_intervention

            # if agrregateannot_officialTitleTarget:
            #     # sanity_counter = sanity_counter + 1
            #     write_hit['aggregate_annot']['official_title_annot'] = agrregateannot_officialTitleTarget
            # if agrregateannot_briefTitleTarget:
            #     # sanity_counter = sanity_counter + 1
            #     write_hit['aggregate_annot']['brief_title_annot'] = agrregateannot_briefTitleTarget
            # if agrregateannot_interventionDescription:
            #     write_hit['aggregate_annot']['intervention_description_annot'] = agrregateannot_interventionDescription # Writes both tokens and annotations
            
            # if agrregateannot_briefSummary:
            #     briefsummary_aggdict = aggregateLongTarget_annot(agrregateannot_briefSummary)
            #     if briefsummary_aggdict:
            #         write_hit['aggregate_annot']['brief_summary_annot'] = briefsummary_aggdict

            # if agrregateannot_detailedDescription:
            #     detailedDescription_aggdict = aggregateLongTarget_annot(agrregateannot_detailedDescription)
            #     if detailedDescription_aggdict:
            #         write_hit['aggregate_annot']['detailed_description_annot'] = detailedDescription_aggdict

        # write the annotation to the annotation file here...
        # Extraction_1_confidence_1
        # if detailedDescription_aggdict or briefsummary_aggdict or interventionDescription_annot or briefTitleTarget_annot or officialTitleTarget_annot:
        #     # Log the information about the NCT ID hit that is already written
        #     logNCTID = 'Writing ID: ' + NCT_id
        #     logging.info(logNCTID)
        #     with open(file_write_trial, 'a+') as wf:
        #         wf.write('\n')
        #         json_str = json.dumps(write_hit)
        #         wf.write(json_str)

    except:
        # Log the information about the NCT ID hit that is already written
        logNCTID = 'Caused exception at the NCT ID: ' + NCT_id
        print(logNCTID)
        # logging.info(logNCTID)

    # print('Total number of synonym matches identified: ', match_scores.count(1.0))

# print(sanity_counter)

# counts = Counter(intervention_types)
# labels, values = zip(*counts.items())
# # sort your values in descending order
# indSort = np.argsort(values)[::-1]

# # rearrange your data
# labels = np.array(labels)[indSort]
# values = np.array(values)[indSort]

# indexes = np.arange(len(labels))

# bar_width = 0.35
# plt.bar(indexes, values)

# # add labels
# plt.xticks(indexes + bar_width, labels)
# plt.xticks(rotation=90)
# plt.savefig('/home/anjani/PICOrecognition/images/interventiontypes_bestconfidence.png', dpi=400, bbox_inches='tight')