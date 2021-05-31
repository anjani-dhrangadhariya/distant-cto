import json, nltk

def getTrainingAnnots(list_NCT):

    count_OOS = 0
    count_entity = 0

    tokenList = []
    tagList = []

    counter_stop = 0
    list_NCT_ids = []

    tokens = []
    labels = []
    nct_ids = []
    length_examiner = []
    with open(list_NCT, 'r', encoding='latin1') as NCT_ids_file:
        next(NCT_ids_file)
        for eachLine in NCT_ids_file:
            annot = json.loads(eachLine)
            id_ = annot['id']

            entity = [] # clean it for each and every new hit
            entity_tags = []

            # Check if aggregate annotations are present in the json
            if 'aggregate_annot' in annot.keys():
                # Read official title
                if 'official_title' in annot['aggregate_annot'].keys():
                    assert len(annot['aggregate_annot']['official_title']) == len(annot['aggregate_annot']['official_title_annot'])
                    raw_labels =  annot['aggregate_annot']['official_title_annot']
                    raw_tokens = annot['aggregate_annot']['official_title']
                    pos_tags = nltk.pos_tag_sents([raw_tokens])
                    stringBuilder = ''
                    tagBuilder = ''
                    for i, (eachToken, eachLabel, eachTag) in  enumerate( zip( raw_tokens, raw_labels, pos_tags[0] ) ):
                        if eachLabel == 1 or eachLabel == 2 or eachLabel == 3 or eachLabel == 4 or eachLabel == 5 or eachLabel == 6 or eachLabel == 7:
                            stringBuilder = stringBuilder + eachToken 
                            tagBuilder = tagBuilder + eachTag[1]
                            count_entity = count_entity + 1
                        elif eachLabel == 0:
                            if stringBuilder:
                                entity.append(stringBuilder)
                            if tagBuilder:
                                entity_tags.append(tagBuilder)
                            stringBuilder = ''
                            tagBuilder = ''
                            count_OOS = count_OOS + 1
                    length_examiner.append( len(annot['aggregate_annot']['official_title_annot']) )
                    nct_ids.append(id_)

                if 'brief_title' in annot['aggregate_annot'].keys():
                    assert len(annot['aggregate_annot']['brief_title']) == len(annot['aggregate_annot']['brief_title_annot'])
                    raw_labels =  annot['aggregate_annot']['brief_title_annot']
                    raw_tokens = annot['aggregate_annot']['brief_title']
                    pos_tags = nltk.pos_tag_sents([raw_tokens])
                    stringBuilder = ''
                    tagBuilder = ''
                    for i, (eachToken, eachLabel, eachTag) in  enumerate( zip( raw_tokens, raw_labels, pos_tags[0] ) ):
                        if eachLabel == 1 or eachLabel == 2 or eachLabel == 3 or eachLabel == 4 or eachLabel == 5 or eachLabel == 6 or eachLabel == 7:
                            stringBuilder = stringBuilder + eachToken
                            tagBuilder = tagBuilder + eachTag[1] 
                            count_entity = count_entity + 1 
                        elif eachLabel == 0:
                            if stringBuilder:
                                entity.append(stringBuilder)
                            if tagBuilder:
                                entity_tags.append(tagBuilder)
                            stringBuilder = ''
                            tagBuilder = ''
                            count_OOS = count_OOS + 1
                    length_examiner.append( len(annot['aggregate_annot']['brief_title_annot']) )
                    nct_ids.append(id_)

                if 'brief_summary_annot' in annot['aggregate_annot'].keys():
                    # iterate the dictionary
                    for eachKey, eachValue in annot['aggregate_annot']['brief_summary_annot'].items():
                        assert len(eachValue) == 2
                        raw_labels =  eachValue[1]
                        raw_tokens = eachValue[0]
                        pos_tags = nltk.pos_tag_sents([raw_tokens])
                        stringBuilder = ''
                        tagBuilder = ''
                        for i, (eachToken, eachLabel, eachTag) in  enumerate( zip( raw_tokens, raw_labels, pos_tags[0] ) ):
                            if eachLabel == 1 or eachLabel == 2 or eachLabel == 3 or eachLabel == 4 or eachLabel == 5 or eachLabel == 6 or eachLabel == 7:
                                stringBuilder = stringBuilder + eachToken 
                                tagBuilder = tagBuilder + eachTag[1]  
                                count_entity = count_entity + 1
                            elif eachLabel == 0:
                                if stringBuilder:
                                    entity.append(stringBuilder)
                                if tagBuilder:
                                    entity_tags.append(tagBuilder)
                                stringBuilder = ''
                                tagBuilder = ''
                                count_OOS = count_OOS + 1
                        length_examiner.append( len( eachValue[1] ) )
                        nct_ids.append(id_)

                if 'detailed_description_annot' in annot['aggregate_annot'].keys():
                    # iterate the dictionary
                    for eachKey, eachValue in annot['aggregate_annot']['detailed_description_annot'].items():
                        assert len(eachValue) == 2
                        raw_labels =  eachValue[1]
                        raw_tokens = eachValue[0]
                        pos_tags = nltk.pos_tag_sents([raw_tokens])
                        stringBuilder = ''
                        tagBuilder = ''
                        for i, (eachToken, eachLabel, eachTag) in  enumerate( zip( raw_tokens, raw_labels, pos_tags[0] ) ):
                            if eachLabel == 1 or eachLabel == 2 or eachLabel == 3 or eachLabel == 4 or eachLabel == 5 or eachLabel == 6 or eachLabel == 7:
                                stringBuilder = stringBuilder + eachToken 
                                tagBuilder = tagBuilder + eachTag[1]  
                                count_entity = count_entity + 1
                            elif eachLabel == 0:
                                if stringBuilder:
                                    entity.append(stringBuilder)
                                if tagBuilder:
                                    entity_tags.append(tagBuilder)
                                stringBuilder = ''
                                tagBuilder = ''
                                count_OOS = count_OOS + 1
                        length_examiner.append( len( eachValue[1] ) )
                        nct_ids.append(id_)

                if 'intervention_description_annot' in annot['aggregate_annot'].keys():
                    # iterate the dictionary
                    for eachKey, eachAnnot in enumerate(annot['aggregate_annot']['intervention_description_annot']):
                        raw_labels =  eachAnnot[1]
                        raw_tokens = eachAnnot[0]
                        pos_tags = nltk.pos_tag_sents([raw_tokens])
                        stringBuilder = ''
                        tagBuilder = ''
                        for i, (eachToken, eachLabel, eachTag) in  enumerate( zip( raw_tokens, raw_labels, pos_tags[0] ) ):
                            if eachLabel == 1 or eachLabel == 2 or eachLabel == 3 or eachLabel == 4 or eachLabel == 5 or eachLabel == 6 or eachLabel == 7:
                                stringBuilder = stringBuilder + eachToken 
                                tagBuilder = tagBuilder + eachTag[1]  
                                count_entity = count_entity + 1
                            elif eachLabel == 0:
                                if stringBuilder:
                                    entity.append(stringBuilder)
                                if tagBuilder:
                                    entity_tags.append(tagBuilder)
                                stringBuilder = ''
                                tagBuilder = ''
                                count_OOS = count_OOS + 1
                        length_examiner.append( len( eachValue[1] ) )
                        nct_ids.append(id_)

            counter_stop = counter_stop + 1
            tokenList.extend( entity )
            tagList.extend( entity_tags )
            # if counter_stop == 20:
            #     break
    return tokenList, tagList, count_entity, count_OOS

def getAnnot(EBMNLP_sentAnnot):
    
    count_OOS = 0
    count_entity = 0

    tokenList = []
    tagList = []
    
    with open(EBMNLP_sentAnnot, 'r', encoding='latin1') as rf:
        for eachAbstract in rf:
            annot = json.loads(eachAbstract)
            abstract_identifier = annot.keys()
            for eachKey in abstract_identifier:
                all_sentences = annot[eachKey]

                for eachSentenceKey in all_sentences.keys():

                    assert len(all_sentences[eachSentenceKey][0]) == len(all_sentences[eachSentenceKey][1])

                    raw_tokens = all_sentences[eachSentenceKey][0]
                    raw_labels = all_sentences[eachSentenceKey][1]
                    pos_tags = nltk.pos_tag_sents([raw_tokens])
                    entity = []
                    entity_tags = []

                    stringBuilder = ''
                    tagBuilder = ''
                    for i, (eachToken, eachLabel, eachTag) in  enumerate( zip( raw_tokens, raw_labels, pos_tags[0] ) ):
                        if eachLabel == '1' or eachLabel == '2' or eachLabel == '3' or eachLabel == '4' or eachLabel == '5' or eachLabel == '6' or eachLabel == '7':
                            stringBuilder = stringBuilder + eachToken 
                            tagBuilder = tagBuilder + eachTag[1]
                            count_entity = count_entity + 1
                        elif eachLabel == '0':
                            if stringBuilder:
                                entity.append(stringBuilder)
                            if tagBuilder:
                                entity_tags.append(tagBuilder)
                            stringBuilder = ''
                            tagBuilder = ''
                            count_OOS = count_OOS + 1
                    tokenList.extend( entity )
                    tagList.extend( entity_tags )


    
    return tokenList, tagList, count_entity, count_OOS

# Get all the validation and test annotations 
EBM_NLP_sentences = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/embnlpgold_sentence_annotation2.txt'
# EBM_NLP_sentences = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/hilfiker_sentence_annotation2.txt'
# EBM_NLP_sentences = '/home/anjani/systematicReviews/data/TA_screening/EBM_NLP/sentence_annotation2.txt'
tokens, tags, count_entities, count_outofspan = getAnnot(EBM_NLP_sentences)
print('Number of entities in the EBM-NLP dataset: ', len(tokens))
print('Number of unique entities in the EBM-NLP dataset: ', len(set(tokens)))

# # get all the training annotatons
list_NCT = '/mnt/nas2/data/systematicReview/clinical_trials_gov/Weak_PICO/intervention_data_preprocessed/extraction1.txt'
training_tokens, training_tags, training_count_entities, training_count_outofspan = getTrainingAnnots(list_NCT)
print('Number of entities in the weak-PICO dataset: ', len(training_tokens))
print('Number of unique entities in the weak-PICO dataset: ', len(set(training_tokens)))

seen_NEs = list(set(training_tokens).intersection(set(tokens)))
print('Total number of seen entities: ', len(seen_NEs))
unseen_NEs = len(set(tokens)) - len(seen_NEs)
print('Total number of unseen entities: ', unseen_NEs)
corpus_diversity = len(seen_NEs) / unseen_NEs
corpus_diversity = corpus_diversity * 100
print('Percentage of unseen entities: ', 100 - corpus_diversity)

seen_Features = list(set(training_tags).intersection(set(tags)))
print('Total number of seen features: ', len(seen_Features))
unseen_features = len(set(tags)) - len(seen_Features)
print('Total number of unseen features: ', unseen_features)
corpus_diversity = len(seen_Features) / unseen_features
corpus_diversity = corpus_diversity * 100
print('Percentage of unseen features: ', 100 - corpus_diversity)

outofspan_total = count_outofspan + training_count_outofspan
entities_total = count_entities + training_count_entities

print('Ratio of entities vs. out of the span tokens: ', (entities_total /  outofspan_total) * 100)