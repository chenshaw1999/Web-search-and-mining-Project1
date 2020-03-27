from pprint import pprint
from Parser import Parser
import util
import numpy as np
class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex={}

    #Tidies terms
    parser=None


    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents]

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString, flag="TF", feedbackSTR=""):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)

        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
        
        if(flag == "feedback"):
            #let words in feedback use the be same tokenize again, prevent from keyerror
            feedbackList = self.parser.tokenise(feedbackSTR)
            for word in feedbackList:    
                #prevent from keyerror again again
                if(word in self.vectorKeywordIndex):
                    vector[self.vectorKeywordIndex[word]] += 0.5

        #if flag == "TF", do nothing
        if(flag == "TF"):
            return vector
        else:
            #calculate the TF-IDF vector
            rawTF = np.array(vector, dtype='f')
            
            rowLength = len(rawTF)
            rowSum = np.sum(rawTF)
            TF = rawTF / rowSum

            n_containing = [sum(num > 0 for num in rawTF)]
            n_containing = np.array(n_containing, dtype='f') 
            n_containing += 1

            IDF = np.full(rowLength, 2048, dtype='f')
            IDF = np.log10(IDF / n_containing)
            TFIDF = TF * IDF
            return TFIDF

    def buildQueryVector(self, termList, flag="TF", feedbackSTR=""):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList), flag, feedbackSTR)
        return query

    def search(self, searchList, flag="TF"):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList, flag)
        
        cosineSimilarity = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        euclideanDistance = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        
        return cosineSimilarity, euclideanDistance

    def pseudoFeedback(self, searchList, flag, feedbackSTR):
        """ search for documents with feedback """
        queryVector = self.buildQueryVector(searchList, flag, feedbackSTR)
        return [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]

