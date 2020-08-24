import nltk
import pandas as pd
from ast import literal_eval
import morfessor

def segmenting(text):
    print("segmentation time BABY!...")
    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file("model.bin")


    sw_text = text
    for i in range(len(sw_text)):
        for h in range(len(sw_text[i])):
          sw_text[i][h] = model.viterbi_segment(sw_text[i][h])[0][0]
    return sw_text
