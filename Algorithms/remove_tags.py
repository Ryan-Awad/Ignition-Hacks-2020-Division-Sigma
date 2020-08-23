def removetags(text): # text is an array for the tokenized words
    if "@" in text[0]:
        text.pop(0)
        text.pop(0)
    
    if "@" in text[-2]:
        text.pop(-1)
        text.pop(-1)

    return text