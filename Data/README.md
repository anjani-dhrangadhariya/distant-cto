# DISTANT-CTO dataset

DISTANT-CTO is a weakly-labeled dataset of 'Intervention' and 'Comparator' entity annotated sentences. The dataset obtained using candidate generation approach described in "DISTANT-CTO: A Zero Cost, Distantly Supervised Approach to Improve Low-Resource Entity Extraction Using Clinical Trials Literature".

## Availability

The dataset is available in `.txt` file format. Full dataset could be downloaded from [the dummy link](https://pages.github.com/).

This directory includes two data files:
1) `extraction1_pos_posnegtrail_conf1.txt` - the `.txt` data files contains all the weak annotations, source intervention terms mapped onto target sentences from clinicaltrials.org (CTO) with a confidence score of 1.0, and
2) `extraction1_pos_posnegtrail_conf09.txt` - the `.txt` data files contains all the weak annotations, source intervention terms mapped onto target sentences from clinicaltrials.org (CTO) with a confidence score of 0.9 and above

## File Structure

The `.txt` data file consists of several lines, each line is stored in a JSON (short for JavaScript Object Notation) object representing one CTO record and the weak annotations obtained from this record. Example line is shown below.

The topmost JSON object from each line consists of 'string:value' pair containing the unique CTO ID of the CTO record (For example 'id:NCT04603443')

There are two nested JSON objects under the root json object with string `'extraction1'` and `'aggregate_annot'`. `'aggregate_annot'` contains all the annotations from `'extraction1'` just in aggregated form. As the project uses `'aggregate_annot'`  for input, it's structure is described below.

![Example JSON file structure](https://github.com/anjani-dhrangadhariya/distant-cto/blob/main/Data/example_file_structure.jpg)

Under the `'aggregate_annot'` JSON object are the 'Intervention' entity-annotated targets *t*. The short targets (comprising only a single sentence) are arranged into an array while the long targets (comprising more than one sentence) are further arranged into a JSON object.

### Description for short targets

Each short target has its own `'targetname'` which is a list of tokens from the tokenized target, `'targetname_annot'` which is a list containing annotation for each individual token from the tokenized target, and `'targetname_pos'` which is a list containing part-of-speech tags for each individual token from the tokenized target. An example of shown below.


![Example](https://github.com/anjani-dhrangadhariya/distant-cto/blob/main/Data/short_annot.jpg)

### Description for long targets

Annotation for each long target in a JSON object is flanked by the `'_annot'` keyword. Each long target which is a paragraph is tokenized into sentences and each sentence is stored as an array under the long target JSON object. Each sentence array is further divided into three lists. List 0 is a list of tokens from the tokenized sentence, list 1 is a list containing annotation for each individual token from the tokenized sentence and list 3 is a list containing part-of-speech tags for each individual token from the tokenized sentence.

An example of shown below.


![Example](https://github.com/anjani-dhrangadhariya/distant-cto/blob/main/Data/short_annot.jpg)