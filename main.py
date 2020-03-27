import os
from VectorSpace import VectorSpace
import argparse
import numpy as np
import nltk

def printResult(searchResult, documentId, weight, measure):
    
    Z = zip(searchResult, documentId)
    
    #in cosine measure we want the bigger one
    if(measure == 'Cosine Similarity'):
        Z = sorted(Z, reverse=True)
    #in euclidean measure we want the smaller one
    else:
        Z = sorted(Z, reverse=False)
    
    #print the current combination, e.g. TF + Sim...
    print('\n' + weight + ' + ' + measure + ':\n')

    print("DocID\t Score")
    
    #only print the top5 results
    for Score, ID in Z[:5]:
        print(ID[:6], '\t',round(Score, 6))

    #return the top ID in method3
    return Z[0][1]


if __name__ == '__main__':

    #using argparse to do this -> python main.py --query "..."
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    args = parser.parse_args()

    #find the document folder loaction
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    docuDir = "/document"
    realDir = fileDir + docuDir

    
    documents = list()
    documentId = list()
    #open all the 2048 documents and append the contents to list
    for file in os.listdir(realDir):
        f = open(realDir + '/' + file, 'r')
        documents.append(f.read())
        documentId.append(file)
    
    vectorSpace = VectorSpace(documents) 
    #print(vectorSpace.vectorKeywordIndex)
    #print(vectorSpace.documentVectors)
    
    #turn query into a list with split(" ")
    query = args.query.split(" ")

    ###################   TF   ####################
    cosineResult, euclideanResult = vectorSpace.search(query)
    
    #print the result
    printResult(cosineResult, documentId, 'TF Weighting', 'Cosine Similarity')
    printResult(euclideanResult, documentId, 'TF Weighting', 'Euclidean Distance')
    
    ################### TF-IDF ####################
    cosineResult, euclideanResult = vectorSpace.search(query, flag="TFIDF")
    
    #method3_top is the ID of the top of TF-IDF + Cosine result document ID
    method3_top = printResult(cosineResult, documentId, 'TF-IDF Weighting', 'Cosine Similarity')
    printResult(euclideanResult, documentId, 'TF-IDF Weighting', 'Euclidean Distance')


    ################### feedback ##################
    
    f = open(realDir + '/' + method3_top, 'r')
    #use nltk library to exract Verb and Noun
    tokens = nltk.word_tokenize(f.read())
    result = nltk.pos_tag(tokens)
    
    document_with_VB_NN = ""
    for i in range(len(result)): 
        #"NN" is the meaning of Noun, "VB" is the meaning of Verb
        if("NN" in result[i][1] or "VB" in result[i][1]):
            #concatenate the NN and VB into one string
            document_with_VB_NN += result[i][0] + " "
    
    #print(document_with_VB_NN)
    
    cos = vectorSpace.pseudoFeedback(query, flag="feedback", feedbackSTR=document_with_VB_NN)
    printResult(cos, documentId, 'Feedback Queries + TF-IDF Weighting', 'Cosine Similarity')
    
    ################################################
