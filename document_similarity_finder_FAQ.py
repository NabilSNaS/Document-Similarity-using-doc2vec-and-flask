import utils
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np


def get_similar_answer_from_csv(faq_path, test_faq, model_path ):
    faq_df = pd.read_csv(faq_path)
    test_faq_df = pd.read_csv(test_faq)
    
    for index, row in test_faq_df.iterrows():
        scores = []
        for index, row_faq in faq_df.iterrows():
            sim_score = utils.similarity_finder(row['Question'], row_faq['Question'], model_path, type="w2v")
            scores.append(sim_score)
        max_score_index = np.argmax(scores)
        answer = faq_df['Answer'][max_score_index]
        print("Question: ", row['Question'])
        print("Answer: ", answer)
        print()
  

def get_similar_answer(faq_path, text_input, model_path ):
    faq_df = pd.read_csv(faq_path)

    scores = []
    for index, row_faq in faq_df.iterrows():
        # print(row_faq['Question'])
        sim_score = utils.similarity_finder(text_input, row_faq['Question'], model_path)
        scores.append(sim_score)
    max_score_index = np.argmax(scores)
    answer = faq_df['Answer'][max_score_index]
    print("Question: ",text_input)
    print("Answer: ", answer)
    print()
  

if __name__ == "__main__":
    faq_path = 'data\FAQs.csv'
    
    test_faq = "data\FAQs_test.csv"

    model_path_d2v = 'models\corpus_word2vec.model'
    get_similar_answer_from_csv(faq_path, test_faq, model_path_d2v )

    # text_input = "What is the date of his death?"
    # get_similar_answer(faq_path, text_input, model_path_d2v )


