import pandas as pd
import nltk
from Algorithms.stopwords import use_stopwords
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

judgement_dataset = pd.read_csv("datasets/contestant_judgment.csv") 

sw_text_list = []
for phrase in judgement_dataset.Text:
    sw_text = use_stopwords(phrase)
    sw_text_list.append(sw_text)
    print("[Cleaned a text]")