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
from sklearn.metrics import f1_score
import pickle

nltk.download('stopwords')

# x is vectorized_phrases and y is training_sentiments
training_df, x, y = preprocessing()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

print("TRAINING...")

clf = SVC(kernel='rbf')
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print("F1_SCORE:", f1_score(y_test, clf.predict(x_test), average=None, labels=[1, 0]))
print("ACCURACY: " + str(accuracy))