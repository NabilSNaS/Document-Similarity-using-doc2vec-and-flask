import utils
from nltk.tokenize import word_tokenize





if __name__ == "__main__":
    text1 = input("Input First Text: ")
    print("\n\n")
    text2 = input("Input Second Text: ")
    #text1 = 'religion atheism Many religions in the world hello fear'
    #text2 = 'islam christian politics abrahamic theism'
    text3 = 'golf golf is a boring game, football is the best'
    #model_path = input("Input Model path: ")
    model_path_d2v = '/home/nsl51/sns/NLP_NER/Assessment/similarity/models/d2v_300_Jun-17-2022'
    model_path_d2v_2 = "/home/nsl51/sns/NLP_NER/Assessment/similarity/models/d2v_Jun-15-2022"
    model_path_w2v = '/home/nsl51/sns/NLP_NER/Assessment/similarity/models/corpus_word2vec.model'
    print("\n\n\nSimilarity Score between two documents: ")
    print(utils.similarity_finder(text1, text2, model_path_d2v_2))
    #print(utils.similarity_finder(text1, text2, model_path_w2v, 'w2v'))