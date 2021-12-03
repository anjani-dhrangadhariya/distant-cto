import urllib.request, json 
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import os
import logging
import datetime as dt
import time

LOG_FILE = os.getcwd() + "/logs"
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)
LOG_FILE = LOG_FILE + "/" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d %H_%M_%S') + 'cto_jsoncrawling' + ".log"
logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
fileHandler = logging.FileHandler("{0}".format(LOG_FILE))
fileHandler.setFormatter(logFormatter)
rootLogger = logging.getLogger()
rootLogger.addHandler(fileHandler)
rootLogger.setLevel(logging.INFO)

readFile = '/mnt/nas2/data/systematicReview/clinical_trials_gov/URLS_CTO/CTOurls_2021.txt'
writeDir = '/mnt/nas2/data/systematicReview/clinical_trials_gov/JSON2021/'

with open(readFile, 'r') as rf:
    for eachNCTID in rf:
        if len(eachNCTID) > 10:
            NCTID = eachNCTID.split('/')[-1]
            full_url = 'https://clinicaltrials.gov/api/query/full_studies?expr=' + NCTID.strip() + '&max_rnk=1&fmt=json'
            print( full_url )
            try:
                with urllib.request.urlopen(full_url) as url:
                    data = json.loads(url.read().decode())

                    writeFile = writeDir + NCTID.strip() + '.json'
                    with open(writeFile, 'w+') as wf:
                        json.dump(data, wf)
                        logging.info(full_url)
            except:
                logNCTID = 'Caused exception at the NCT ID: ' + NCTID
                logging.info(logNCTID)