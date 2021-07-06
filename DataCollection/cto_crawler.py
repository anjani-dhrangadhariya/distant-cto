## Scrape CTO
## Crawl CTO: https://clinicaltrials.gov/ct2/about-site/crawling

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import os
import logging
import datetime as dt
import time

LOG_FILE = os.getcwd() + "/logs"
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)
LOG_FILE = LOG_FILE + "/" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d %H_%M_%S') + 'cto_basecrawling' + ".log"
logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
fileHandler = logging.FileHandler("{0}".format(LOG_FILE))
fileHandler.setFormatter(logFormatter)
rootLogger = logging.getLogger()
rootLogger.addHandler(fileHandler)
rootLogger.setLevel(logging.INFO)

class CTOCrawler():

    def linkcrawler(link, pattern):
        
        crawl_domain = 'https://clinicaltrials.gov'
        
        urls = []
        
        res = requests.get(link).text
        soup = BeautifulSoup(res,"lxml")
        links = soup.find_all('a')
        urls_all = [x['href'] for x in links]
        for x in urls_all:
            if pattern in x:
                full_link = crawl_domain + x
                urls.append(full_link)
        
        return urls

startURL = 'https://clinicaltrials.gov/ct2/about-site/crawling'
baseURLs = CTOCrawler.linkcrawler(startURL, '/ct2/crawl/')


outputfilename = '/home/anjani/PICOrecognition/CTOurls.txt'

# Get all the further links that require crawling
for eachbaseURL in baseURLs:
    print(eachbaseURL)
    furtherURLs = CTOCrawler.linkcrawler(eachbaseURL, '/ct2/show/')
      
    # write to the output file
    with open(outputfilename, 'a+') as outfile:
        outfile.write("\n".join(furtherURLs))  
        outfile.write("\n")
        logging.info(eachbaseURL)