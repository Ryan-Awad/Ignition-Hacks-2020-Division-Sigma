# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:57:52 2020

@author: Robbot
"""
# import these modules 
import nltk
import pandas as pd
from ast import literal_eval
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk import wordnet

def lemoning(text):
    lemmatizer = WordNetLemmatizer() 
    print("lemmatization time BABY!...")
    sw_text = text
    for i in range(len(text)):
        for h in range(len(text[i])):
            sw_text[i][h] = lemmatizer.lemmatize(sw_text[i][h])
    return sw_text      
