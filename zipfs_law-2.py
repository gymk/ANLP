# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:43:25 2019

@author: yuvaraja manikandan
"""

# Zipf Law
# https://www.cs.swarthmore.edu/~richardw/classes/cs65/f18/lab01.html for loglog

# https://www.researchgate.net/publication/221200132_Exploring_Regularity_in_Source_Code_Software_Science_and_Zipf's_Law

from operator import itemgetter
import nltk
# from nltk.probability import FreqDist
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import math

stop_words = set(stopwords.words('english'))

words = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
#words = nltk.Text(nltk.corpus.gutenberg.words('carroll-alice.txt'))

# convert to lower case and consider only words
words = [word.lower() for word in words]  #if word.isalpha()
# remove stop words
# words = [word for word in words if word not in stop_words]

# =============================================================================
# fDist = FreqDist(words[:2000])
# 
# print(fDist.B())
# print(fDist.Nr(1153))
# 
# #for k,v in fDist.items():
#     #print(k, v)
#  
# fDist.plot()
# =============================================================================

# build word frequency as dictionary
frequency = {}
for word in words:
    count = frequency.get(word, 0)
    frequency[word] = count + 1
 
# sort and create a list
freq_list = sorted(frequency.items(), key=itemgetter(1), reverse=True)
#print(type(freq_list), freq_list)

# plot loglog graph of rank versus frequency
ranks = range(1, len(freq_list)+1) # x-axis: ranks
freqs = [freq for (word, freq) in freq_list] # y-axis: frequencies

print('50th Elem Freq: ', freqs[50])
print('150th Elem Freq: ', freqs[150])
print('150th*3 ==> ', freqs[150] * 3)

plt.loglog(ranks, freqs, label='austen-emma.txt')
plt.xlabel('log(rank)')
plt.ylabel('log(freq)')
plt.legend(loc='lower left')

# plot zip law's expected value
# https://www.cs.swarthmore.edu/~richardw/classes/cs65/f18/lab01.html
def H_approx(n):
    """
    Returns an approximate value of n-th harmonic number
    http://en.wikipedia.org/wiki/Harmonic-number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

T = len(words)
print('T: ', T)
vocab = set([word for (word, freq) in freq_list])
n = len(vocab)
print('n: ', n)
k = T/H_approx(n)
print('k: ', k)
expected_freq = [k/r for r in ranks]
plt.loglog(ranks, expected_freq, label="zipf's law")
plt.legend(loc='lower left')

plt.show()
