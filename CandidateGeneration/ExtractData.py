#!/usr/bin/env python
'''
This module contains functions that extract source and target regions from the input CTO trial study. 
'''
__author__ = "Anjani Dhrangadhariya"
__maintainer__ = "Anjani Dhrangadhariya"
__email__ = "anjani.k.dhrangadhariya@gmail.com"
__status__ = "Prototype/Research"

################################################################################
# Functions to wade through CTO files
################################################################################

################################################################################
# --------------------------------    Sources   --------------------------------
################################################################################
def getInterventionSource(protocol_section):
    # Check if the protocol section has interventions list
    if 'ArmsInterventionsModule' in protocol_section:
        if 'InterventionList' in protocol_section['ArmsInterventionsModule']:
            if 'Intervention' in protocol_section['ArmsInterventionsModule']['InterventionList']:
                intervention = protocol_section['ArmsInterventionsModule']['InterventionList']['Intervention']
                for eachIntervention in intervention:
                    yield eachIntervention

def getInterventionNames(protocol_section):
    interventions = []
    intervention_syn = []

    # Check if the protocol section has interventions list
    if 'ArmsInterventionsModule' in protocol_section:
        if 'InterventionList' in protocol_section['ArmsInterventionsModule']:
            if 'Intervention' in protocol_section['ArmsInterventionsModule']['InterventionList']:
                intervention = protocol_section['ArmsInterventionsModule']['InterventionList']['Intervention']
                for eachIntervention in intervention:
                    if 'InterventionName' in eachIntervention:
                        interventions.append( eachIntervention['InterventionName'] )
                    if 'InterventionOtherName' in eachIntervention:
                        intervention_syn.append( eachIntervention['InterventionOtherName'] )

    return interventions, intervention_syn

def getArmsGroups(protocol_section):
    armgrouplabels = []
    # Check if the protocol section has arm groups list
    if 'ArmsInterventionsModule' in protocol_section:
        if 'ArmGroupList' in protocol_section['ArmsInterventionsModule']:
            if 'ArmGroup' in protocol_section['ArmsInterventionsModule']['ArmGroupList']:
                armGroup = protocol_section['ArmsInterventionsModule']['ArmGroupList']['ArmGroup']
                for eachArmGroup in armGroup:
                    if 'ArmGroupLabel' in eachArmGroup:
                        armgrouplabels.append(eachArmGroup['ArmGroupLabel'])

    return armgrouplabels

################################################################################
# --------------------------------    Targets   --------------------------------
################################################################################

def getBriefSummary(protocol_section):

    # Check if the protocol section has Description Module
    if 'DescriptionModule' in protocol_section:
        if 'BriefSummary' in protocol_section['DescriptionModule']:
            briefSummary = protocol_section['DescriptionModule']['BriefSummary']
            return briefSummary

def getDetailedDescription(protocol_section):

    # Check if the protocol section has Description Module
    if 'DescriptionModule' in protocol_section:
        if 'DetailedDescription' in protocol_section['DescriptionModule']:
            detailedDescription = protocol_section['DescriptionModule']['DetailedDescription']
            return detailedDescription

def getOfficialTitle(protocol_section):

    # Check if the protocol section has Identification Module
    if 'IdentificationModule' in protocol_section:
        if 'OfficialTitle' in protocol_section['IdentificationModule']:
            OfficialTitle = protocol_section['IdentificationModule']['OfficialTitle']
            return OfficialTitle

def getBriefTitle(protocol_section):

    # Check if the protocol section has Identification Module
    if 'IdentificationModule' in protocol_section:
        if 'OfficialTitle' in protocol_section['IdentificationModule']:
            BriefTitle = protocol_section['IdentificationModule']['BriefTitle']
            return BriefTitle