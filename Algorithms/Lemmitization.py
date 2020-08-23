# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:57:52 2020

@author: Robbot
"""


# import these modules 
from Algorithms.stopwords import use_stopwords
import nltk
import pandas as pd
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
        
    print(sw_text_list[0][0])

    for i in range(len(sw_text_list)):
        for h in range(len(sw_text_list[i])):
            sw_text_list[i][h] = lemmatizer.lemmatize(sw_text_list[i][h])
