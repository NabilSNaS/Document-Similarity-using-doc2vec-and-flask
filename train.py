from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from datetime import datetime
import os
import sys
import numpy as np
import utils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import nltk
nltk.download('punkt')
from datetime import date



from gensim.models import Word2Vec



###################  train ###############


def trainDoc2Vec(TEXT_DATA_DIR, vector_size = 100, alpha = 0.025, min_count = 1, epochs = 10, save = False):
    '''
    takes train data directory, vector size, alpha(learning rate), min_count, epochs, save(binary- > if want to save the model)

    returns the trained model
    '''
    texts,  labels_index, labels = utils.preprocess(TEXT_DATA_DIR)
    tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(texts)]
    model_d2v = Doc2Vec(vector_size= vector_size, epochs=epochs, alpha= alpha, min_count= min_count)


    model_d2v.build_vocab(tagged_data)

    #for epoch in range(epochs):
    #   print("Epoch  " + str(epoch) + " running .... ")
    model_d2v.train(tagged_data, total_examples=model_d2v.corpus_count, epochs = model_d2v.epochs)
    print("Training Completed!")
    if save is True:
        now = datetime.now()
        today = date.today()
        d = today.strftime("%b-%d-%Y")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        model_d2v.save("models/d2v_300_" + d)

    return model_d2v 
        
    #most_similar(10000,pairwise_similarities,'Cosine Similarity', top = 10)
    #most_similar(1000,pairwise_differences,'Euclidean Distance', top = 5)



def train_word2vec(TEXT_DATA_DIR, vector_size=300, window=10, min_count=5, workers=11, alpha=0.025, epochs=20):
    '''
    takes word2vec model parameters and data path as input

    returns trained model
    '''

    texts,  labels_index, labels = utils.preprocess(TEXT_DATA_DIR)
    corpus_sentences = [word_tokenize(doc) for doc in texts ]
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, alpha=alpha, epochs=epochs)

    model.build_vocab(corpus_sentences)
    model.train(corpus_sentences, total_examples=model.corpus_count, epochs=model.epochs)

    model.save('models/corpus_word2vec.model')




if __name__ == "__main__":
    #/home/nsl51/sns/NLP_NER/Assessment/similarity/data/20news-bydate-train
    TEXT_DATA_DIR = "data/20news-bydate-train"

    # d2v_model = trainDoc2Vec(TEXT_DATA_DIR, vector_size = 100, alpha = 0.025, min_count = 1, epochs=250,  save = True)
    w2v_model = train_word2vec(TEXT_DATA_DIR, vector_size = 100, epochs = 100)
    #/home/nsl51/sns/NLP_NER/Assesment/20_newsgroup/20_newsgroup
    