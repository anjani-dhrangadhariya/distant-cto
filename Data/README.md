# DISTANT-CTO dataset

DISTANT-CTO is a weakly-labeled dataset of 'Intervention' and 'Comparator' entity annotated sentences. The dataset obtained using candidate generation approach described in "DISTANT-CTO: A Zero Cost, Distantly Supervised Approach to Improve Low-Resource Entity Extraction Using Clinical Trials Literature".

## Availability

The dataset is available in `.txt` file format. Full dataset could be downloaded from [here](https://pages.github.com/).

## File Structure

The `.txt` data file consists of several lines, each line is stored in a `JSON` (short for JavaScript Object Notation) object representing one CTO record and the weak annotations obtained from this record. Example line is shown below.

The topmost `JSON` object from each line consists of 'string:value' pair containing the unique CTO ID of the CTO record (For example 'id:NCT04603443')

There are two nested JSON objects under the root json object with string `'extraction1'` and `'aggregate_annot'`. `'aggregate_annot'` contains all the annotations from 'extraction1' just in aggregated form. As the project uses 'aggregate_annot' for input, it's structure is described below.