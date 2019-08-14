# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:39:22 2019

@author: yuvaraja manikandan
"""

"""
Rank the documents with respect to the query q=[0.1 0.001 0.0 0.2]Tusing cosine similarity using cosine similarity

                 D1  D2  D3
car              0.1 0.0 0.1
comprehensive    0.1 0.1 0.1
third-party      0.0 0.9 0.0
insurance        0.1 0.9 0.02
"""

import numpy as np

weight_matrix = np.array([
        [0.1, 0.0, 0.1],
        [0.1, 0.1, 0.1],
        [0.0, 0.9, 0.0],
        [0.1, 0.9, 0.02]])

q = np.array([0.1, 0.001, 0.0, 0.2])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


#print(cosine_similarity(weight_matrix[:,0].T, q.T))

(term_count, doc_count) = weight_matrix.shape

for term_index in range(doc_count):
    print('D{0}'.format(term_index+1), cosine_similarity(weight_matrix[:,term_index], q))
