
def mergeOldNew(merge_to, ds_annot):

    merged_annot = [0] * len(ds_annot)
    for counter, (o_a, n_a) in enumerate(zip( merge_to, ds_annot  )):
        chosen_annot = max( int(o_a), int(n_a) )
        merged_annot[counter] = chosen_annot

    return merged_annot