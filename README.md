# `DISTANT-CTO`

### DISTANT-CTO: A Zero Cost, Distantly Supervised Approach to Improve Low-Resource Entity Extraction Using Clinical Trials Literature

PICO recognition is an information extraction task for identifying Participant, Intervention, Comparator, and Outcome (PICO elements) information from clinical literature. Manually identifying PICO elements is the most time-consuming and cognitively demanding step for conducting systematic reviews (SR) that form the basis of evidence-based clinical practice. Deep learning approaches can identify these entities from a set of clinical texts in minutes which can otherwise take months per SR to complete. However, the lack of large, annotated corpora restricts innovation and adoption of automated PICO recognition systems. The largest-available PICO corpus is a manually annotated dataset that involved hiring and training medical students and physicians, which is too expensive for a majority of the scientific community.
To break through this bottleneck, we propose a novel distant supervision approach, DISTANT-CTO, or distantly supervised PICO extraction using the clinical trials literature, to generate a massive weakly-labeled dataset with 977,682 high-quality 'Intervention' and 'Comparator' entity annotations. We use our insights to train distant NER (named-entity recognition) models using this weakly-labeled dataset and demonstrate that it outperforms even the sophisticated models trained on the manually annotated dataset with a 2\% F1 improvement over the I and C entities of the PICO benchmark and more than 5\% improvement when combined with the manually annotated dataset.
We demonstrate the generalizability of our approach for under-represented, non-pharmaceutical entities by gaining an impressive F1-score on another domain-specific PICO benchmark. The approach is not only zero-cost but is also adaptable and scalable for a constant stream of PICO annotations.

