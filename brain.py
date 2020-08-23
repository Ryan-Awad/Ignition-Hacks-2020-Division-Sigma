import pandas as pd
import nltk
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from Algorithms.stopwords import use_stopwords
from Algorithms.preprocessing import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

nltk.download('stopwords')

# x is vectorized_phrases and y is training_sentiments
training_df, x, y = preprocessing()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

print("TRAINING...")
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)

print("ACCURACY: " + str(accuracy))


# -----------------------------------------------

# df = pd.read_csv("datasets/training_data.csv")
#
# x_train, x_test, y_train, y_test = train_test_split(df.Text,df.Sentiment,test_size=0.33,random_state=42) #splitting the dataset into training and testing
#
# #check the dimensions
# print("Number of training samples: {}".format(x_train.shape[0]))
# print("Number of testing samples: {}".format(x_test.shape[0]))
#
# #bag of words model to convert text to numbers
# cv=CountVectorizer(binary=False)
# print(type(cv))
# print(cv)
#
# #transformed train reviews
# train_reviews = cv.fit_transform(x_train)
# #transformed test reviews
# test_reviews = cv.transform(x_test)
# print(type(train_reviews))
# print(x_train)
#
# #labeling the sentient data
# lb=LabelBinarizer()
# #transformed sentiment data
# sentiment_data = lb.fit_transform(y_train)
# sentiment_test = lb.fit_transform(y_test)
# print(sentiment_data.shape)
#
# #The data preparation is done, steps below prepare a sample model
# #preparing the model
# lr=LogisticRegression()
# #training the model for Bag of words
# lr_bow=lr.fit(train_reviews,sentiment_data)
#
# #Accuracy score for bag of words
# lr_bow_predict = lr.predict(test_reviews)
# lr_bow_score = accuracy_score(sentiment_test,lr_bow_predict)
# print("lr_bow_score :",lr_bow_score)
