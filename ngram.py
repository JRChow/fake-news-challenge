import re
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity

english_stemmer = nltk.stem.SnowballStemmer('english')
token_pattern = r"(?u)\b\w\w+\b"
stopwords = set(nltk.corpus.stopwords.words('english'))

def ngram(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n+1):
        output.append(input[i:i+n])
    return output


def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

def preprocess_data(line, token_pattern=token_pattern, exclude_stopword=True, stem=True):
    token_pattern = re.compile(token_pattern, flags=re.UNICODE)
    tokens = [x.lower() for x in token_pattern.findall(line)]
    tokens_stemmed = tokens
    if stem:
        tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]

    return tokens_stemmed

def getUnigram(words):
    return words

def getBigram(words,skip = 0):
    L = len(words)
    if L>1:
        lst = []
        for i in range(L-1):
            for k in range(1, skip+2):
                if i+k < L:
                    lst.append(' '.join([words[i], words[i+k]]))
    else:
        lst = getUnigram(words)
    return lst

def getTrigram(words, skip = 0):
    L = len(words)
    if L>2:
        lst = []
        for i in range(L-2):
            for k1 in range(1, skip+2):
                for k2 in range(1, skip+2):
                    if i+k1 < L and i+k1+k2 < L:
                        lst.append(' '.join([words[i], words[i+k1], words[i+k1+k2]]))
    else:
        lst = getBigram(words, skip)
    return lst