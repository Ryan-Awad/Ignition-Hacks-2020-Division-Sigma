import pandas as pd
from Algorithms.stopwords import use_stopwords

judgement_dataset = pd.read_csv("datasets/contestant_judgment.csv")
stopwords = use_stopwords(judgement_dataset)

