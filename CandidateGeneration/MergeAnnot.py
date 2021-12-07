
import collections

def partition(lst, n):
    """Return successive n-sized chunks from list (lst)."""
    chunks = []
    for i in range(0, len(lst), n):
        chunks.append( lst[i:i + n]  )
    return chunks

def mergeAnnotations(main_dict_merged, key_t, value_t, annot_type, input_annot):

    if key_t not in main_dict_merged:
        main_dict_merged[key_t] = value_t

        if annot_type not in main_dict_merged[key_t]:
            main_dict_merged[key_t][annot_type] = input_annot
        else:
            merge_to = main_dict_merged[key_t][annot_type]
            merged_annot = [0] * len(input_annot)
            assert len(merge_to) == len(input_annot) == len(value_t['tokens'])

            for counter, (o_a, n_a) in enumerate(zip( merge_to, input_annot )):
                chosen_annot = max( int(o_a), int(n_a) )
                merged_annot[counter] = chosen_annot

            main_dict_merged[key_t][annot_type] = merged_annot

    elif key_t in main_dict_merged:
        if annot_type in main_dict_merged[key_t]:

            merge_to = main_dict_merged[key_t][annot_type]
            merged_annot = [0] * len(input_annot)
            assert len(merge_to) == len(input_annot) == len(value_t['tokens'])

            for counter, (o_a, n_a) in enumerate(zip( merge_to, input_annot )):
                chosen_annot = max( int(o_a), int(n_a) )
                merged_annot[counter] = chosen_annot

            main_dict_merged[key_t][annot_type] = merged_annot

        elif annot_type not in main_dict_merged[key_t]:
            main_dict_merged[key_t][annot_type] = input_annot

    return main_dict_merged

# aggregate the annotations here
def aggregateLongTarget_annot(agrregateannot_briefSummary):
    """Aggregate annotations from multiple intervention sources for each target."""
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

def pos_neg_trail(aggregated_dictionary):
    """Generate and return +- trailing annotations."""
    values_ = []
    for key, value in aggregated_dictionary.items():
        values_.extend( value[1] )

    mergedChunks_dictionary = dict()
    
    if 1 in values_:
        # Sort the dictionary
        aggregated_dictionary_sorted = collections.OrderedDict(sorted(aggregated_dictionary.items()))

        # Partition into chunks
        chunks = partition( list(aggregated_dictionary_sorted.keys()), 4)

        # Merge each chunk into a single key-value pair
        for eachChunk in chunks:
            keyChunks = []
            valueChunk_sent = []
            valueChunk_lab = []
            valueChunk_pos = []
            for eachChunk_i in eachChunk:
                keyChunks.append( eachChunk_i )
                valueChunk_sent.extend( aggregated_dictionary_sorted[eachChunk_i][0] )
                valueChunk_lab.extend( aggregated_dictionary_sorted[eachChunk_i][1] )
                valueChunk_pos.extend( aggregated_dictionary_sorted[eachChunk_i][2] )

            assert len(valueChunk_sent) == len(valueChunk_lab) == len(valueChunk_pos)

            mergedKey = str('_'.join(keyChunks))
            mergedChunks_dictionary[mergedKey] = [valueChunk_sent, valueChunk_lab, valueChunk_pos]

    if bool(mergedChunks_dictionary) == True:
        return mergedChunks_dictionary