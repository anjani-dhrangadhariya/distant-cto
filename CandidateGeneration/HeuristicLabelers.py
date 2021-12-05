import re
import numpy as np

def find_indices(lst, compiled_regex):
    return [i for i, elem in enumerate(lst) if compiled_regex.findall( elem ) ]

def regexMatcher(target):

    annot = [0] * len(target['tokens'])

    inter_regex = '([tT]herapy|[tT]reatment)'
    compiled_regex = re.compile(inter_regex)
    
    if re.search( compiled_regex, target['text'] ): # TODO: POS tags

        regex_matches = re.findall(compiled_regex, target['text'])
        exact_matches = list(filter(compiled_regex.findall, target['tokens']))
        match_indices = find_indices(target['tokens'], compiled_regex)

        for i, e_m, r_m in zip(match_indices, exact_matches, regex_matches):
            
            if len(e_m) > len(r_m) and target['pos'][i] in ['PROPN', 'NOUN']:
                annot[i] = 1

            range_from_start = list( range(i, -1, -1) )
            for r in range_from_start:
                if target['pos'][r] in ['PROPN', 'NOUN']:
                    annot[r] = 1
                else:
                    break

    assert len(annot) == len(target['tokens'])
    return target['tokens'], annot