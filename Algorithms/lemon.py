import nltk
import pandas as pd
from ast import literal_eval
from nltk.stem import WordNetLemmatizer 
from nltk import wordnet

def lemonade(text):
    lemmatizer = WordNetLemmatizer() 

    lemon_text = []
    for t in text:
        lemon_text.append(lemmatizer.lemmatize(t))
    
    return lemon_text