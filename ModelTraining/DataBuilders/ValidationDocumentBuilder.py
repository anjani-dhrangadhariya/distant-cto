import json 
import pandas as pd

def labelGenerator(raw_labels):

    bio_labels = []
    for i, eachLabel in enumerate(raw_labels):
        if eachLabel == '0':
            bio_labels.append(0)
        elif raw_labels[i] == '1' and raw_labels[i-1] == '0':
            bio_labels.append(1)
        elif raw_labels[i] == '1' and raw_labels[i-1] == '1':
            bio_labels.append(2)  

    return bio_labels

def readEBMNLP_docAnnot(EBMNLP_docAnnot, label_type):

    print('Getting the sentence level annotations frm EBM-NLP....')

    tokenList = []
    labelList = []
    posList = []
    abstractIdentifiersList = []
    
    with open(EBMNLP_docAnnot, 'r', encoding='latin1') as rf:
        for eachAbstract in rf:
            annot = json.loads(eachAbstract)
            abstract_identifier = annot.keys()
            for eachKey in abstract_identifier:
                
                all_sentences = annot[eachKey]

                tokens = all_sentences[0]
                annotations = all_sentences[1]
                if label_type == 'BIO':
                    annotations = labelGenerator(annotations)
                pos = all_sentences[2]

                tokenList.append( tokens )
                labelList.append( annotations )
                posList.append( pos )

    assert len(tokenList) == len(labelList)
    # Collate the lists into a dataframe
    corpusDf = percentile_list = pd.DataFrame({'tokens': tokenList, 'labels': labelList, 'pos': posList})
    df = corpusDf.sample(frac=1).reset_index(drop=True) # Shuffles the dataframe after creation

    return df