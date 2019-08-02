# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:16:16 2019

@author: yuvaraja manikandan
"""

from operator import itemgetter
import nltk
import pandas as pd

frequency = {}
words_emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))

for word in words_emma:
    count = frequency.get(word, 0)
    frequency[word] = count + 1
    
rank = 1
column_header = ['Rank', 'Frequency', 'Frequency * Rank']
df = pd.DataFrame(columns=column_header)

for word, freq in reversed(sorted(frequency.items(), key=itemgetter(1))):
    df.loc[word] = [rank, freq, rank * freq]
    rank = rank + 1
    
print(df)

