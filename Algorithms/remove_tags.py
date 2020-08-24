from nltk.tokenize import word_tokenize

def removetags(text): # text is an array for the tokenized words
    text = word_tokenize(text)

    if "@" in text[0]:
        text.pop(0)
        text.pop(0)

    return text