# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:20:22 2019

@author: yuvaraja manikandan
"""

"""
A scientist extracted a table from a research paper that listed
three molecular fingerprints in the binary form as given below:
    
    1. [0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,0]
    2. [0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1]
    3. [0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,0]
    
He wants to find the similarity of the molecules using some similarity measure.
He found that he could use Tanimoto coefficient to find the similarity.
The formula for Tanimoto Coefficient is

T_c(A,B) = c/(a+b-c)

where
    c is the length of A intersection B
    a and b are sizes of A and B respectively.
    
Use the description to answers question 5-7

Tanimoto Coefficient: https://www.surechembl.org/knowledgebase/84207-tanimoto-coefficient-and-fingerprint-generation
"""

fg = [
        [0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,0],
        [0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,0]
        ]
    
def getFingerPrintOnCount(A):
    """
    Input:
        Iterable having boolean values 0 or 1
    Output:
        Count of number of 1 found in A
    """
    count = 0
    for i in A:
        if(i == 1):
            count += 1
    return count

def getTanimotoCoefficient(A, B):
    """
    Input:
        A: Fingerprint of Query Structure
        B: Fingerprint of Target Structure
    Output:
        TanimotoCoefficient similarity score of A w.r.to B
    """
    intersection = []
    for index in range(len(A)):
        val = A[index] & B[index]
        if(val == 1):
            intersection.append(val)
    a = getFingerPrintOnCount(A)
    b = getFingerPrintOnCount(B)
    c = getFingerPrintOnCount(intersection)
    #print(a, b, c)
    return float(c/(a + b - c))

for ai, A in enumerate(fg):
    for bi, B in enumerate(fg):
        #if(A != B):
        print('Tanimoto Coefficient Similarity of D{0} with D{1}'.format(ai+1, bi+1), getTanimotoCoefficient(A, B))
    print(' ')