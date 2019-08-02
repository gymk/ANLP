# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:26:03 2019

@author: yuvaraja manikandan
"""

# Zipf Law

from operator import itemgetter
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))

words = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))

# convert to lower case
words = [word.lower() for word in words if word.isalpha()]
# remove stop wrods
words = [word for word in words if word not in stop_words]

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


frequency = {}
 
for word in words:
    count = frequency.get(word, 0)
    frequency[word] = count + 1
 
freq_list = sorted(frequency.items(), key=itemgetter(1), reverse=True)

# =============================================================================
# limit_val = 0
# for k, v in freq_list:
#     print(k, v)
#     limit_val = limit_val + 1
#     if limit_val > 100:
#         break
#      
# =============================================================================

freq_list = freq_list[:20]
index = range(len(freq_list))
l1 = list(zip(*freq_list))

# plot frequency list
plt.bar(index, l1[1])
plt.xlabel('Words')
plt.ylabel('Counts')
plt.xticks(index, l1[0], fontsize=15, rotation=30)
plt.title('High Frequency Words')
plt.show()


# plot rank
rank = 1
rank_list = []
for freq in l1[1]:
    rank_list.append(rank * freq)
    rank = rank + 1
    
#index
print(rank_list)
plt.bar(index, rank_list)
plt.xlabel('Words')
plt.ylabel('Counts')
plt.xticks(index, l1[0], fontsize=15, rotation=30)
plt.title('High Frequency Words')
plt.show()

