# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:55:40 2019

@author: yuvaraja manikandan
"""

import nltk
from nltk.corpus import stopwords
import pandas as pd

stop_words = stopwords.words('english')

def getHeapLawValues(corpus_name):
    words = nltk.Text(nltk.corpus.gutenberg.words(corpus_name))
    # normalize the words
    words = [w.lower() for w in words if w.isalpha()]
    # remove stop words
    words = [w for w in words if w not in stop_words]
    
    M = len(set(words))
    T = len(words)
    
    print(corpus_name, ' M: ', M, ' T: ', T, 'Ratio: ', M/T)
    return [M, T, M/T]

column_headers = ['M', 'T', 'Ratio']
df = pd.DataFrame(columns=column_headers)

for corpus in nltk.corpus.gutenberg.fileids():
    df.loc[corpus] = getHeapLawValues(corpus)
    
print(df)
#df.plot()
df.Ratio.plot()