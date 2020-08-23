from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def use_stopwords(text):
    sw = set(stopwords.words('english'))
    text_tokenized = word_tokenize(text.lower())
    text_sw = [word for word in text_tokenized if word not in sw]

    return text_sw