# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:33:54 2020

@author: Robbot
"""
import morfessor
import pickle
model_file = "model.bin"


bruh_file_write = open("segment.txt", 'wb')
bruh_file_read = open("segment.txt", 'rb')
pickle.dump(model_file, bruh_file_write)
bruh_file_write.close()
lmao = pickle.load(bruh_file_read)






model_file = "model.bin"
io = morfessor.MorfessorIO()
model = io.read_binary_model_file(model_file)

word = "running"
# for segmenting new words we use the viterbi_segment(compound) method
print(model.viterbi_segment(word)[0][0])