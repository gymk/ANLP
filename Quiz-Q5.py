# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:53:55 2019

@author: yuvaraja manikandan
"""

import numpy as np

docs = {'TF1' : 25/127, 'TF2' : 3/250, 'TF3' : 20/650, 'TF9' : 15/125, 'TF1000': 20/800}
TF = [25/127, 3/250, 20/650, 15/125, 20/800]

for k,v in docs.items():
    print(k, v)
    
idf = np.log10(100000/200)
print("IDF: ", idf)

print('Rank:')
for k,v in docs.items():
    docs[k] = v*idf

#for x in TF:
    #print('TF-IDF: ', x*idf)
    
for k,v in docs.items():
    print(k, v)
    
print('Sorted:')
for k,v in sorted(docs.items(), key=itemgetter(1)):
    print(k, v)
    