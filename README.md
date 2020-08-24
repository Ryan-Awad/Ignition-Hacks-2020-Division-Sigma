# Ignition Hacks 2020 Division Sigma

In this code we used sklearn’s svm to create an AI to determine whether a given statement was positive or negative. 

Beginning with a csv file containing the user, ID, Review and its sentiment. With this, we stripped it of irrelevant data (including Users and IDs), preprocessed our reviews (which includes filtering sentences to only contain relevant and using bag of words to transfer words into binary). Applying this preprocessed data, we used the “Sklearn SVM” model (using an RBF kernel) to record trends and create a working AI. This AI was tested with an accuracy test and an F1_score (the F1_score was not in the main code but was used during development). Finally, we used our new AI on the attached “Judgement Data” to predict the sentiment of real world messages.

This project was composed of members: Anusha Shekhar, Michelle Pansa, Robert Saab, Ryan Awad

<h3>Dependencies:</h3>

 - Pandas==1.0.5
 - NLTK==3.5
 - SKLearn==0.23.2
 - Morfessor==2.0.6
