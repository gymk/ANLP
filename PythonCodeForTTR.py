# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:35:41 2019

@author: yuvaraja manikandan
"""

import nltk
from nltk.corpus import stopwords

# get the stop words for English
stop_words = set(stopwords.words('english'))

# get the Corpus
words_bryant = nltk.Text(nltk.corpus.gutenberg.words('bryant-stories.txt'))
words_emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))

# convert to small letters
words_bryant = [word.lower() for word in words_bryant if word.isalpha()]
words_emma = [word.lower() for word in words_emma if word.isalpha()]

# remove stop words
# Length of both bryant and austen emma texts vary
# For TTR we can't compare corpus having various length
# So we will limit to 15000 words
words_bryant = [word for word in words_bryant if word not in stop_words][:15000]
words_emma = [word for word in words_emma if word not in stop_words][:15000]

# Calculate Type-Token Ratio
TTR_Bryant = len(set(words_bryant))/len(words_bryant)
TTR_Emma = len(set(words_emma))/len(words_emma)

print('Number of token, vocabulary, Type-Toekn Ration (Bryant Stories) = ',
      len(words_bryant), len(set(words_bryant)), TTR_Bryant)

print('Number of token, vocabulary, Type-Toekn Ration (Emma Stories) = ',
      len(words_emma), len(set(words_emma)), TTR_Emma)