import nltk
import pandas as pd
from ast import literal_eval
import morfessor
from Algorithms.stopwords import use_stopwords


nltk.download('punkt')
training_df = pd.read_csv("training_data.csv")
bruh_file_read = open("segment.txt", 'rb')
model_file = pickle.load(bruh_file_read)


io = morfessor.MorfessorIO()
model = io.read_binary_model_file("model.bin")

def segmenting():
    print("segmentation time BABY!...")

    sw_text_list = []
    for phrase in training_df.Text:
        sw_text = use_stopwords(phrase)
        sw_text_list.append(sw_text)

        
    for i in range(len(sw_text_list)):
        for h in range(len(sw_text_list[i])):
          sw_text_list[i][h] = model.viterbi_segment(sw_text_list[i][h])[0][0]
