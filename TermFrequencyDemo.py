# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:56:12 2019

@author: yuvaraja manikandan
"""

import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

#read the corpus
words = nltk.Text(nltk.corpus.gutenberg.words('bryant-stories.txt'))
#convert to small letters
words = [word.lower() for word in words if word.isalpha()]
words = [word for word in words if word not in stop_words]

fDist = FreqDist(words)

print(len(words)) # 21718
print(len(set(words))) # 3688 - unique words

# Let's see top 10 common words
for x,v in fDist.most_common(10):
    print(x, v)
    
# Add Weights by normalizing the values
for x,v in fDist.most_common(10):
    print(x, v/len(fDist))