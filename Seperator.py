# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import csv

# Here we will take the first four results to use as test data
with open ("training_data.csv") as file:
    reader = csv.reader(file)
    g = []
# These are the variables we will be using
    answers = []
    reviews = []
    count = 0
    for row in reader:
        count += 1
        if count >= 5:
            break
        g.append(row)
# Here we will add will split the sample reviews into answers and reviews
    for i in range(len(g)):
        reviews.append(g[i][2])
        answers.append(g[i][3])        
