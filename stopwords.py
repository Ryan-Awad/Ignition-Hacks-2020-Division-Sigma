import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import os

nltk.download('punkt')
nltk.download('stopwords')

os.system('cls')

stopwords = set(stopwords.words('english'))
phrase_list = pd.read_csv("contestant_judgment.csv", sep=',')
phrase_cleaned_list = []

cleaned_phrase_num = 0 # remove

for phrase in phrase_list.Text:
    cleaned_phrase_num += 1 # remove
    print("Cleaned phrase [" + str(cleaned_phrase_num) + "/" + str(len(phrase_list.Text)) + "]") # remove
    phrase_tokenized = word_tokenize(phrase.lower())
    phrase_cleaned = [word for word in phrase_tokenized if word not in stopwords]
    phrase_cleaned_list.append(phrase_cleaned)

print(phrase_cleaned_list) # remove