import pandas as pd
import nltk
from Algorithms.stopwords import use_stopwords
from Algorithms.preprocessing import *

nltk.download('punkt')
nltk.download('stopwords')

sw_text_list = []
for phrase in training_df.Text:
    sw_text = use_stopwords(phrase)
    sw_text_list.append(sw_text)
    # print(sw_text)

preprocessing()
