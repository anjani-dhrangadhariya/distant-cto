# All the identifiers that require to be scraped could be found in the file: /home/anjani/PICOrecognition/CTOurls.txt

import os
import logging
import datetime as dt
import time

urlInputFilename = '/home/anjani/PICOrecognition/CTOurls.txt'
urlOutputFilename = '/home/anjani/PICOrecognition/CTO_identifiersurls.txt'
baseXMLurl = 'https://clinicaltrials.gov/ct2/show/'

with open(urlInputFilename, 'r') as cto_url:
    for eachLine in cto_url:
            
        trial_identifier = eachLine.split('/')[-1]
        baseXMLurl_identifier = str(baseXMLurl).rstrip() + str(trial_identifier).rstrip() + str('?displayxml=true').rstrip()

        # write to the output file
        with open(urlOutputFilename, 'a+') as outfile:
            outfile.write(baseXMLurl_identifier)
            outfile.write("\n")
            logging.info(baseXMLurl_identifier)