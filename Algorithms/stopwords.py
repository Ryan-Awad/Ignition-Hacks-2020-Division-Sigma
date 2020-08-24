from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def use_stopwords(text):
    sw = set(stopwords.words('english'))
    text_sw = [word for word in text if word not in sw]

    return text_sw