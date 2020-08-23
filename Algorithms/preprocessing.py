import nltk
import re
import numpy as np
import pandas as pd
import heapq
from ast import literal_eval
from Algorithms.stopwords import use_stopwords

nltk.download('punkt')
training_df = pd.read_csv("datasets/training_data.csv")

# training_df = training_df[:10000]

def preprocessing():
    print(training_df)
    print("PREPROCESSING...")

    sw_text_list = []
    for phrase in training_df.Text:
        sw_text = use_stopwords(phrase)
        sw_text_list.append(sw_text)
        # print(sw_text)

    training_df['sw_text'] = sw_text_list
    training_df['sw_text'] = training_df['sw_text'].str.lower().replace(regex=r'\W', value=' ').replace(regex=r'\s+', value=' ').replace(regex=r'\d+', value=' ')

    vectorized_phrases = []
    training_sentiments = []
    bow = {}

    print("CREATING BOW...")
    for ind, row in training_df.iterrows():
        training_sentiments.append(row["Sentiment"])

        # create dictionary to contain bag of words from the tokenized list
        for word in str(row['sw_text']):
            if word not in bow.keys():
                bow[word] = 1
            else:
                bow[word] += 1

    print("VECTORIZING...")
    # take the top 20 words with the highest frequency from each phrase
    freq_words = heapq.nlargest(10000, bow, key=bow.get)

    #assign a vector to each frequent word
    freq_arr = []
    for phrase in training_df['sw_text']:
        vector = []
        for word in freq_words:
            # if word in nltk.word_tokenize(phrase):
            if word in str(phrase):
                vector.append(1)
            else:
                vector.append(0)
        freq_arr.append(vector)
    vectorized_phrases = np.asarray(freq_arr)

    # vectorized_phrases.append(freq_arr)

    vectorized_phrases = np.asarray(vectorized_phrases)

    # returns original read dataframe, array of sentiments, array of vectorized phrases
    return training_df, vectorized_phrases, training_sentiments
