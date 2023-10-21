import os
import sys
import numpy as np
import gensim.models as g

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


from Docsim import DocSim

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

import nltk
nltk.download('punkt')
from datetime import date
 

from sklearn.preprocessing import MinMaxScaler



def most_similar(doc_id,texts, labels, labels_index, model ,matrix, top = None):
    '''
    Takes document id, text(whole text data), labels(labels of the documents), label index(dictionary of labels and label id) , model(doc2vec model) , matrix(cosine or euclidean distance based similarity), top(how many similar documents needed to see)
    prints most similar documents of the given document id
    '''
    document_embeddings=np.zeros((len(texts),100))
    for i in range(len(document_embeddings)):
        document_embeddings[i]=model.docvecs[i]
        
        
    pairwise_similarities=cosine_similarity(document_embeddings)
    pairwise_differences=euclidean_distances(document_embeddings)



    print (f'ID : {doc_id},  Document topic: {get_key(labels[ix],labels_index)}, Document: {texts[doc_id]}')
    print ('\n')
    print ('Similar Documents:')
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(pairwise_similarities[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(pairwise_differences[doc_id])
         
    i = 0
    for ix in similar_ix:
        if top is not None:
            if i == top:
                break
        if ix==doc_id:
            continue
        print('\n')
        print("ID: ", ix)
        print(f'Document topic: {get_key(labels[ix],labels_index)}')
        print (f'Document: {texts[ix]}')
       # print (f'{matrix} : {similarity_matrix[doc_id][ix]}')
        i += 1


def get_key(val, my_dict):
    '''
    takes value
    get key from  value of a dictionary
    returns key
   
    '''

    for key, value in my_dict.items():
        if val == value:
             return key
 
    return None
 
def preprocess(TEXT_DATA_DIR):
    '''
    takes data directory path as string

    returns list of documents
    '''
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)

    return texts,  labels_index, labels 


def evaluation_data_maker(df):
    '''
    takes pandas dataframe

    returns a dataframe with normalized similarity, predicted class, ground truth columns and also returns  accuracy from the evaluation
    '''
    df['gt'] = df['text1_class'] == df['text2_class']
    df['gt'] = df['gt'].map({True: 1, False: 0})
    df['norm_similarity'] = MinMaxScaler().fit_transform(np.array(df['similarity']).reshape(-1,1))
    temp = df['norm_similarity'] > 0.45  #threshold 0.45
    df['predicted_class'] = temp.map({True: 1, False: 0})
    acc = len(df[df['predicted_class'] == df['gt']])/len(df)
    return df, acc



def similarity_finder(text1, text2, model_path, type='d2v'):
    '''
    takes 2 documents and a doc2vec model as input
    returns similarity between the documents
    '''
    if type == 'd2v':
        model = g.Doc2Vec.load(model_path)
        sim = model.similarity_unseen_docs(word_tokenize(text1), word_tokenize(text2))

    elif type== 'w2v':      
        model = Word2Vec.load(model_path)
        ds = DocSim(model.wv)
        sim_scores = ds.calculate_similarity(text1, [text2])
        sim = list(sim_scores[0].values())[0]
        
    return sim




