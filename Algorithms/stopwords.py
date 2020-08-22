import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import os

def use_stopwords(judgement_dataset):
    nltk.download('punkt')
    nltk.download('stopwords')
    os.system('cls')

    sw = set(stopwords.words('english'))
    phrase_sw_list = []
    cleaned_phrase_num = 0 # remove

    for phrase in judgement_dataset.Text:
        cleaned_phrase_num += 1 # remove
        print("Cleaned phrase [" + str(cleaned_phrase_num) + "/" + str(len(judgement_dataset.Text)) + "]") # remove

        phrase_tokenized = word_tokenize(phrase.lower())
        phrase_sw = [word for word in phrase_tokenized if word not in sw]
        phrase_sw_list.append(phrase_sw)

    return phrase_sw_list
