import os
import pandas as pd

def readBIOESLabels(train_label_files, train_label_dir, tag_map):

    labels = dict()

    for each_file in train_label_files:

        with open(os.path.join(train_label_dir, each_file), 'r') as f:
            sentence_labels_encoded = []
            for token in f.read().splitlines():
                if token == '00':
                    token = '0'
                sentence_labels_encoded.append(tag_map[token])

            # Convert normal labels to BIO labels
            bio_labels = []
            for i, eachLabel in enumerate(sentence_labels_encoded):
                if eachLabel == 0:
                    bio_labels.append(0)
                elif sentence_labels_encoded[i] == 1 and sentence_labels_encoded[i-1] == 0:
                    bio_labels.append(1)
                elif sentence_labels_encoded[i] == 1 and sentence_labels_encoded[i-1] == 1:
                    bio_labels.append(2) 

            labels[each_file.split('.')[0]] = bio_labels # Each filename is assigned the token labels here
    train_label_files_ = [each_file.split('.')[0] for each_file in train_label_files]
    return labels, train_label_files_


def get_vocab_and_tag_maps(code):

    ##################################################################################
    # Import word vocabulary
    ##################################################################################
    words_path = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/words.txt'
    vocab = {}
    with open(words_path) as f:
        for i, l in enumerate(f.read().splitlines()):
            vocab[l] = i

    # Import tags vocabulary 
    if code == 'P':
        tags_path = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/tagsP.txt'
    if code == 'I':
        tags_path = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/tagsI.txt'
    if code == 'O':
        tags_path = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/tagsO.txt'

    tag_map = dict()
    with open(tags_path) as f:
        for i, l in enumerate(f.read().splitlines()):
            tag_map[l] = i
    
    return vocab, tag_map

def readFineGrainedLabels(train_label_files, train_label_dir):
    labels = dict()

    for each_file in train_label_files:
        with open(os.path.join(train_label_dir, each_file), 'r') as f:
            sentence_labels_encoded = []
            sentence_labels_encoded = [0 if token=='O' else int(token) for token in f.read().splitlines() ]
            labels[each_file.split('.')[0]] = sentence_labels_encoded
    train_label_files_ = [each_file.split('.')[0] for each_file in train_label_files]
    return labels, train_label_files_

def readLabels(train_label_files, train_label_dir, tag_map):

    labels = dict()

    for each_file in train_label_files:
        with open(os.path.join(train_label_dir, each_file), 'r') as f:
            sentence_labels_encoded = []
            for token in f.read().splitlines():
                if token == '00':
                    token = '0'
                sentence_labels_encoded.append(tag_map[token])
            labels[each_file.split('.')[0]] = sentence_labels_encoded
    train_label_files_ = [each_file.split('.')[0] for each_file in train_label_files]
    return labels, train_label_files_

def getLabels(tag_map, code, experiment_type):

    if code == 'P':
        train_label_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/starting_spans/participants/annot/train'
        test_label_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/starting_spans/participants/annot/test/gold'
        test_label_dir_hilfiker = '/home/anjani/systematicReviews/data/TA_screening/hilfiker_sr_ta/PICO_annotation_project/validation_files/labels/participants/annot'

        train_label_fine_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/train/'
        test_label_fine_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/test/gold/'
        test_label_fine_dir_hilfiker = '/home/anjani/systematicReviews/data/TA_screening/hilfiker_sr_ta_fine/PICO_annotation_project/validation_files/labels/participant/'

    elif code == 'I':
        train_label_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/starting_spans/interventions/train'
        test_label_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/starting_spans/interventions/test/gold'
        test_label_dir_hilfiker = '/home/anjani/systematicReviews/data/TA_screening/hilfiker_sr_ta/PICO_annotation_project/validation_files/labels/intervention/annot'

        train_label_fine_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/interventions/train/'
        test_label_fine_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/interventions/test/gold/'
        test_label_fine_dir_hilfiker = '/home/anjani/systematicReviews/data/TA_screening/hilfiker_sr_ta_fine/PICO_annotation_project/validation_files/labels/intervention/'

    elif code == 'O':
        train_label_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/starting_spans/outcomes/train'
        test_label_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/starting_spans/outcomes/test/gold'
        test_label_dir_hilfiker = '/home/anjani/systematicReviews/data/TA_screening/hilfiker_sr_ta/PICO_annotation_project/validation_files/labels/outcome/annot'

        train_label_fine_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/outcomes/train/'
        test_label_fine_dir = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/outcomes/test/gold/'
        test_label_fine_dir_hilfiker = '/home/anjani/systematicReviews/data/TA_screening/hilfiker_sr_ta_fine/PICO_annotation_project/validation_files/labels/outcome/'

    else:
        raise "Provide one of these codes (P, I, O)" 
    
    train_label_files = os.listdir(train_label_dir)
    if experiment_type == 'BIO':
        train_encoded_labels, train_files = readBIOESLabels(train_label_files, train_label_dir, tag_map)
    elif experiment_type == 'seq_lab':
        train_encoded_labels, train_files = readLabels(train_label_files, train_label_dir, tag_map)
        #train_encoded_fine_labels, train_files = readFineGrainedLabels(train_label_files, train_label_fine_dir)


    test_label_files = os.listdir(test_label_dir)
    if experiment_type == 'BIO':
        test_encoded_labels, test_files = readBIOESLabels(test_label_files, test_label_dir, tag_map)
    elif experiment_type == 'seq_lab':
        test_encoded_labels, test_files = readLabels(test_label_files, test_label_dir, tag_map)
        #test_encoded_fine_labels, test_files = readFineGrainedLabels(test_label_files, test_label_fine_dir)


    test_label_files_hilfiker = os.listdir(test_label_dir_hilfiker)
    if experiment_type == 'BIO':
        test_encoded_labels_hilfiker, test_files_hilfiker = readBIOESLabels(test_label_files_hilfiker, test_label_dir_hilfiker, tag_map)
    elif experiment_type == 'seq_lab':
        test_encoded_labels_hilfiker, test_files_hilfiker = readLabels(test_label_files_hilfiker, test_label_dir_hilfiker, tag_map)
        #test_encoded_fine_labels_hilfiker, test_files_hilfiker = readFineGrainedLabels(test_label_files_hilfiker, test_label_fine_dir_hilfiker)

    return train_encoded_labels, test_encoded_labels, test_encoded_labels_hilfiker, train_files


def get_data_loaders(EBM_NLP_texts, experiment_type):

    # Get the data (titles and abstracts from training and test set) tokens 
    train_abstracts_encoded = dict()
    test_abstracts_encoded = dict()

    vocab, tag_map = get_vocab_and_tag_maps('I')

    # Get the span labels
    print('Fetching fine grained labels')
    train_labels, test_labels, test2_labels, train_label_files_  = getLabels(tag_map, 'I', experiment_type)

    # Load the development and test sets
    # Load texts for EBM-NLP
    data_branches = [ name for name in os.listdir(EBM_NLP_texts) if os.path.isdir(os.path.join(EBM_NLP_texts, name)) ]
    for each_dir in data_branches:
        print(each_dir)
        files = os.listdir(os.path.join(EBM_NLP_texts, each_dir))
        for each_file in files:
            if each_file.endswith('.tokens'):
                with open(os.path.join(EBM_NLP_texts, each_dir, each_file), 'r') as f:
                    sentence = []
                    sentence = [ token for token in f.read().splitlines() ]
                    if 'train' in each_dir and each_file.split('.')[0] in train_label_files_:
                        train_abstracts_encoded[each_file.split('.')[0]] = sentence
                    elif 'test' in each_dir :
                        test_abstracts_encoded[each_file.split('.')[0]] = sentence
    print('Loading the abstracts and the abstract labels completed...')


    test2_abstracts_encoded = dict()
    hilfiker_dir = '/home/anjani/systematicReviews/data/TA_screening/hilfiker_sr_ta/PICO_annotation_project/validation_files/tokens'
    hilfiker_files = os.listdir(hilfiker_dir)
    for each_file in hilfiker_files:
        with open(os.path.join(hilfiker_dir, each_file), 'r') as f:
            sentence = []
            sentence = [ token for token in f.read().splitlines() ]           
            test2_abstracts_encoded[each_file.split('.')[0]] = sentence

    assert len(test2_abstracts_encoded) == len(test2_labels)
    print('Loading test tokens from Hilfiker et al. completed...')

    train_keys = list(train_abstracts_encoded.keys())
    train_keys = train_keys[0:10]

    test_keys = list(test_abstracts_encoded.keys())
    test2_keys = list(test2_abstracts_encoded.keys())

    # Combine text tokens and labels into a dataframe
    tokenized_train = []
    tokenized_train_labels = []
    for eachKey in train_keys:
        if eachKey in train_abstracts_encoded and eachKey in train_labels:
            tokenized_train.append(train_abstracts_encoded[eachKey])
            tokenized_train_labels.append(train_labels[eachKey])

    # XXX Test set 1
    tokenized_test = []
    tokenized_test_labels = []
    for eachKey in test_keys:
        if eachKey in test_abstracts_encoded and eachKey in test_labels:
            tokenized_test.append( test_abstracts_encoded[eachKey] )
            tokenized_test_labels.append( test_labels[eachKey] )
    print('Retrieved test (EBM-NLP) coarse label set of size: ', len(tokenized_test_labels))

    # XXX Test set 2
    tokenized_test2 = []
    tokenized_test2_labels = []
    for eachKey in test2_keys:
        if eachKey in test2_abstracts_encoded and eachKey in test2_labels:
            tokenized_test2.append( test2_abstracts_encoded[eachKey] )
            tokenized_test2_labels.append( test2_labels[eachKey] )
    print('Retrieved test (hilfiker et al.) coarse label set of size: ', len(tokenized_test2_labels))
    
    # Collate the lists into a dataframe
    testDf = percentile_list = pd.DataFrame({'tokens': tokenized_train,'labels': tokenized_train_labels})
    test1Df = percentile_list = pd.DataFrame({'tokens': tokenized_test,'labels': tokenized_test_labels})
    test2Df = percentile_list = pd.DataFrame({'tokens': tokenized_test2,'labels': tokenized_test2_labels})
    
    return testDf, test1Df, test2Df