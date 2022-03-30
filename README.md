# `DISTANT-CTO` approach

DISTANT-CTO combines methods from distant supervision and dynamic programming and uses freely-available resources like [clinicaltrial.org](https://clinicaltrials.gov/) to obtain a massive corpus of 'Intervention' and 'Comparator' entity annotations.

- Candidate generation: The process of generating pseudo-labeled or distant-labeled dataset using the combination of distant supervision and dynamic programming. We call these distantly labeled dataset as DISTANT-CTO.
- Model training: Once the distantly-labeled candidates are generated, transformer-based discriminative 'Intervention' and 'Comparator' NER models were trained. 

<img src="https://github.com/anjani-dhrangadhariya/distant-cto/blob/main/Data/candidategenerationcolor_cmyk.jpeg" width="640"/>
