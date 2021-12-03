'''
Main python file for reading candidates generated during the candidate generation phase
'''
__author__ = "Anjani Dhrangadhariya"
__maintainer__ = "Anjani Dhrangadhariya"
__email__ = "anjani.k.dhrangadhariya@gmail.com"
__status__ = "Prototype/Research"

##################################################################################
# Imports
##################################################################################
# staple imports

# Read the annotations into a dataframe
import gensim
import numpy as np

# keras essentials
from keras.preprocessing.sequence import pad_sequences

from torchtext.legacy import data
import torchtext.vocab as vocab


def createAttnMask(input_ids, k):
    # Add attention masks
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int( token_id.sum(axis=0) != (k*2) ) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return np.array(attention_masks)

def choose_vector_type(vector_type):

    if vector_type == 'word2vec':
        word2vec_path = "/mnt/nas2/data/Personal/Anjani/wordEmbeddings/GoogleNews-vectors-negative300.bin" # Google news vectors
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=20000)
        return word2vec, 300
    if vector_type == 'bio2vec_2':
        bio2vec2_path = "/mnt/nas2/data/Personal/Anjani/wordEmbeddings/bio_nlp_vec/PubMed-shuffle-win-2.bin"
        bio2vec2 = gensim.models.KeyedVectors.load_word2vec_format(bio2vec2_path, binary=True, limit=20000)
        return bio2vec2, 200
    if vector_type == 'bio2vec_30':
        bio2vec30_path = "/mnt/nas2/data/Personal/Anjani/wordEmbeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin"
        bio2vec30 = gensim.models.KeyedVectors.load_word2vec_format(bio2vec30_path, binary=True, limit=20000)
        return bio2vec30, 200
    else:
        print('Provide the correct vector type information.')


def get_average_word2vec(tokens_list, vector, k, generate_missing=False):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        # vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
        vectorized = [vector[word] if word in vector else np.random.uniform(0.00001, 10**(-30), size=k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    return vectorized

def get_word2vec_embeddings(vectors, clean_questions, k, generate_missing=False):
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors, k, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

def get_semantic_vectors(annotations_df, vector_type, MAX_LEN):

    # chosen_vector, k = choose_vector_type(vector_type)
    # embeddings = get_word2vec_embeddings(chosen_vector, annotations_df, k = k)

    # # get vectors
    # embeddings = pad_sequences( annotations_df , maxlen=MAX_LEN, value=2, padding="post", dtype='float32') # padding the sequence with 2 because 0 can be a word vector

    # # Generate masks
    # attention_masks = createAttnMask(embeddings, k = k)

    # # Pad the labels
    # labels = pad_sequences( annotations_df['labels'] , maxlen=MAX_LEN, value=0, padding="post")
    # # labels = labels.tolist()

    # return embeddings.tolist(), labels, attention_masks, vector_type

    chosen_vector, k = choose_vector_type(vector_type)

    # use torchtext to define the dataset field containing text
    text_field = data.Field(sequential=True, use_vocab=True, lower=True)
    label_field = data.Field(sequential=True, use_vocab=False, lower=False)
    fields=[ ('text', text_field), ('label', label_field) ]

    # load your dataset using torchtext, e.g.
    dataset = ds.SequenceTaggingDataset(examples=annotations_df, fields=fields )

    print( dataset[0] )
    

    return dataset, chosen_vector