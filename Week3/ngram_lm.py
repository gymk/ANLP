from collections import defaultdict
from nltk import bigrams
from nltk import trigrams
from pprint import pprint

"""
Module for building Langugae Models
using Bigram and Trigram
"""

class NGramLM:
    def __init__(self, corpus, ss=True, es=True, start_stop_count=1):
        """
        corpus: List having sentences as documents
        ss : flag to include start symbol <S> in the document
        es: flag to include end symbol </S> in the document
        start_stop_count: Number of start, stop symbols to add in each document
        
        OOV not considered yet
        
        This module requires following modules to be imported
            from collections import defaultdict
            from nltk import bigrams
            from nltk import trigrams
        """
        self.ss = ss
        self.es = es
        self.start_stop_count = start_stop_count
        self.corpus = corpus
        self.printCorpus()
        self.printTokens()
        self.ngrams_model = self.getDefaultDic()
        
    def getDefaultDic(self):
        return defaultdict(lambda: defaultdict(lambda: 0))
        
    def printCorpus(self):
        """
        Dumps corpus content
        """
        for sent in self.corpus:
            print(sent)
            
    def printTokens(self):
        """
        Dumps tokenized corpus (after pre-processing)
        """
        for i, sent in enumerate(self.corpus):
            tokens = self.preProcessSentence(sent)
            print(i, tokens)
                
                
    def printBiGramModel(self):
        for term_1 in self.ngrams_model:
            print('For ' + term_1 +':')
            for term_2 in self.ngrams_model[term_1]:
                print('\t' + term_1, term_2, self.ngrams_model[term_1][term_2])
                
    def printTrigramModel(self):
        for term_1 in self.ngrams_model:
            term_1_str = ' '.join(list(term_1))
            print('For ' + term_1_str +':')
            for term_2 in self.ngrams_model[term_1]:
                print('\t' + term_1_str, term_2, self.ngrams_model[term_1][term_2])
        
    def preProcessSentence(self, sent):
        """
        pre process the given sentence
        and returns it as a list of tokens
        """
        tokens = [word.lower() for word in sent.split() if word.isalpha()]
        if self.ss:
            for i in range(self.start_stop_count):
                tokens.insert(0, "<S>")
        if self.es:
            for i in range(self.start_stop_count):
                tokens.append("<\S>")
        return tokens
    
    def computePMF(self):
        """
        Computes Probability Distribution Mass for all n-grams
        starting with first term
        """
        # compute the probability for the bigram starting with w1
        for term1 in self.ngrams_model:
            # total count of bigrams starting with w1
            bigram_count_for_term1 = float(sum(self.ngrams_model[term1].values()))
            print('Count: ', bigram_count_for_term1)
            # distribute the probability mass for all bigrams starting with w1
            for term2 in self.ngrams_model[term1]:
                self.ngrams_model[term1][term2] /= bigram_count_for_term1
        
    def buildBiGramModel(self):
        """
        Constructs bigram model using the corpus
        """
        
        print("\n Building Bigram Model ...")
        self.ngrams_model = self.getDefaultDic()
        
        # Build bigram model
        for sent in self.corpus:
            tokens = self.preProcessSentence(sent)
            for w1,w2 in bigrams(tokens):
                self.ngrams_model[w1][w2] += 1
        self.printBiGramModel()
        
        # compute the probability for the bigram starting with w1
        self.computePMF()
                
        #self.printBiGramModel()
    
    def buildTriGramModel(self):
        """
        Constructs trigram model using the corpus
        """
        
        print("\n Building Trigram Model ...")
        self.ngrams_model = self.getDefaultDic()
        
        # Build bigram model
        for sent in self.corpus:
            tokens = self.preProcessSentence(sent)
            print(tokens)
            for w1,w2,w3 in trigrams(tokens):
                self.ngrams_model[(w1, w2)][w3] += 1
        #pprint(self.ngrams_model)
        
        # compute the probability for the bigram starting with w1
        self.computePMF()
                
        self.printTrigramModel()
    
    def getBiGramProbability(self, doc):
        """
        Returns the probability score using the built model
        """
        
        # Build the bigram model
        tokens = self.preProcessSentence(doc)
        
        val = 1
        for w1,w2 in bigrams(tokens):
            print('Query Bi ', w1, w2, self.ngrams_model[w1][w2])
            val *= self.ngrams_model[w1][w2]

        return val
        
    def getTriGramProbability(self, doc):
        """
        Returns the probability score using the built model
        """
        
        # Build the bigram model
        tokens = self.preProcessSentence(doc)
        
        val = 1
        for w1,w2,w3 in trigrams(tokens):
            print('Query Tri ', w1, w2, w3, self.ngrams_model[(w1, w2)][w3])
            val *= self.ngrams_model[(w1, w2)][w3]

        return val