import nltk
import re
import numpy as np
import pandas as pd
import heapq

nltk.download('punkt')

training_df = pd.read_csv("datasets/test.csv")
print(training_df)

for ind in range(len(training_df.index)):
    text = training_df.loc[ind,"Text"] #str

    ## tokenize each review
    tokenized_phrase = nltk.sent_tokenize(text) #list

    for i in range(len(tokenized_phrase)): #tokenized_phrase[i] is str
        tokenized_phrase[i] = tokenized_phrase[i].lower()
        tokenized_phrase[i] = re.sub(r'\W', ' ', tokenized_phrase[i])
        tokenized_phrase[i] = re.sub(r'\s+', ' ', tokenized_phrase[i])
        tokenized_phrase[i] = re.sub(r'\d+', ' ', tokenized_phrase[i])

    # create bow from the tokenized list
    bow = {} #dictionary to contain bag of words

    for phrase in tokenized_phrase:
        words = nltk.word_tokenize(phrase)
        for word in words:
            if word not in bow.keys():
                bow[word] = 1
            else:
                bow[word] += 1

    #take the top 10 words with the highest frequency from each phrase
    freq_words = heapq.nlargest(10, bow, key=bow.get)

    #assign a vector to each frequent word
    freq_arr = []
    for phrase in tokenized_phrase:
        vector = []
        for word in freq_words:
            if word in nltk.word_tokenize(phrase):
                vector.append(1)
            else:
                vector.append(0)
        freq_arr.append(vector)
    freq_arr = np.asarray(freq_arr)

    print(freq_arr)
