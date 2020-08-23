# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:57:52 2020

@author: wadid
"""
# import these modules 
import nltk
import re
import numpy as np
import pandas as pd
import heapq
from ast import literal_eval
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk import wordnet

nltk.download('punkt')
training_df = pd.read_csv("Sheet1.csv")
#print(training_df)

lemmatizer = WordNetLemmatizer() 


def lemoning():
    print("lemmatization time BABY!...")

    sw_text_list = []
    for phrase in training_df.Text:
        sw_text = use_stopwords(phrase)
        sw_text_list.append(sw_text)
    for i in range(len(sw_text)):
        sw_text[i] = lemmatizer.lemmatize(sw_text[i]) 
        print(sw_text[i])






lemoning()
