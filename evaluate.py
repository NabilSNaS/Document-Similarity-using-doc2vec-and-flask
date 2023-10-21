import utils
from datetime import datetime
from datetime import date

from nltk.tokenize import word_tokenize
import pandas as pd

def evaluate(test_data_path,   model_path, type='d2v'):
    '''
    Takes test data path, model path as input

    Prints test data similarity between the test data( between same class and different class as well)
    generates csv file with the evaluation of data
    provides accuracy
    '''
   

    texts,  labels_index, labels = utils.preprocess(test_data_path)
    comparison_Type = []	
    text1_index	 = [] 
    text1_class	= [] 
    text2_index	= []
    text2_class	 = [] 
    similarity = []

   
    for i in range(0,len(texts), 5):
        for j in range(1, 5):
            print("Similarity(same_class) between index " + str(i) + " ( " + utils.get_key(labels[i],labels_index) + " ) " + " and " + str(i + j) + " ( " + utils.get_key(labels[i + j],labels_index) + " ) " + " :  " +  str(utils.similarity_finder(texts[i], texts[i + j], model_path, type)))
            comparison_Type.append("Similarity(same_class)")	
            text1_index.append(i) 
            text1_class.append(utils.get_key(labels[i],labels_index))
            text2_index.append(i+ j)
            text2_class.append(utils.get_key(labels[i + j],labels_index)) 
            similarity.append(utils.similarity_finder(texts[i], texts[i + j], model_path, type))

        for j in range(0,len(texts), 5):
            if  i == j:
                continue
            print("Similarity(different_class) between index " + str(i) + " ( " + utils.get_key(labels[i],labels_index) + " ) " + " and " + str(j) + " ( " + utils.get_key(labels[j],labels_index) + " ) " + " :  " +  str(utils.similarity_finder(texts[i], texts[j], model_path, type)))
            comparison_Type.append("Similarity(different_class)")	
            text1_index.append(i) 
            text1_class.append(utils.get_key(labels[i],labels_index))
            text2_index.append(j)
            text2_class.append(utils.get_key(labels[j],labels_index)) 
            similarity.append(utils.similarity_finder(texts[i], texts[j], model_path, type))

        if i + 5 > len(texts):
            break
    

    df = pd.DataFrame(list(zip(comparison_Type,text1_index, text1_class, text2_index, text2_class, similarity)),
               columns =['comparison_Type','text1_index', 'text1_class', 'text2_index', 'text2_class', 'similarity'])
    df,  acc  = utils.evaluation_data_maker(df)
    now = datetime.now()
    today = date.today()
    d = today.strftime("%b-%d-%Y")
    #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    df.to_csv("output_" + type + "_" + d + ".csv")

    print("Accuracy is : ", acc )

    

if __name__ == "__main__":
    model_path = '/home/nsl51/sns/NLP_NER/Assessment/similarity/models/d2v_300_Jun-17-2022'
    test_data_path = "/home/nsl51/sns/NLP_NER/Assessment/similarity/data/20news-bydate-test2"

    evaluate(test_data_path,   model_path, 'd2v')