
def mergeOldNew(main_dict, key_t, annot_type, ds_annot):

    if annot_type not in main_dict[key_t]['annot']:
        main_dict[key_t]['annot']['ds'] = ds_annot
    else:
        merge_to = main_dict[key_t]['annot']['ds']
        merged_annot = [0] * len(ds_annot)
        for counter, (o_a, n_a) in enumerate(zip( merge_to, ds_annot  )):
            chosen_annot = max( int(o_a), int(n_a) )
            merged_annot[counter] = chosen_annot

    return main_dict