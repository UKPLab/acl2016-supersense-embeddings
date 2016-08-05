# -*- coding: utf-8 -*-

import re
from unidecode import unidecode
import numpy as np
from keras.preprocessing import sequence

"""
Functions to read in the text and supersense files, 
create suitable numpy matrices for train/dev/test

@author: Lucie Flekova
"""

#file contains two tab-separated columns: 1) space separated plain text, 2) space separated supersense text
def readFile(filepath, posit):
    sentences = []
    maxlen = 0
    maxlen2 = 0    
    for line in open(filepath,'r'):
        line = line.strip().split('\t')
        if (maxlen<line[0].split()): maxlen=len(line[0].split())
        if (maxlen2<line[1].split()): maxlen2=len(line[1].split())
        sentences.append([line[0], line[1], posit]) 
    return sentences, maxlen, maxlen2



#function to map words and supersenses to their embedding indices         
def createNumpyArray(sentences, maxleng, maxleng2, word2Idx,  sim2Idx, freq2Idx, label2Idx):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']  
    unknownSimIdx = sim2Idx['UNKNOWN']
    paddingSimIdx = sim2Idx['PADDING'] 
    unknownFreqIdx = freq2Idx['UNKNOWN']
    paddingFreqIdx = freq2Idx['PADDING']   
    
    xMatrix = [] #plain text embedding indices row by row
    simMatrix = [] #plain text supersense similarity vector indices
    freqMatrix = [] #plain text supersense frequency vector indices
    sMatrix = [] #supersense text embedding indices row by row
    yVector = [] #outcome labels (1,0)
    
    wordCount = 0
    unknownWordCount = 0
    unknownSCount = 0
    unknownSimCount = 0
    unknownFreqCount = 0

    for sentence in sentences:
        wordIndices = []
        simIndices = []    
        freqIndices = []
        senseIndices = []

        for word in sentence[0].split(): #plain text, space separated
                
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()] 
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1

                if word in sim2Idx:
                    simIdx = sim2Idx[word]
                elif word.lower() in sim2Idx:
                    simIdx = sim2Idx[word.lower()] 
                else:
                    simIdx = unknownSimIdx
                    unknownSimCount += 1

                if word in freq2Idx:
                    freqIdx = freq2Idx[word]
                elif word.lower() in freq2Idx:
                    freqIdx = freq2Idx[word.lower()] 
                else:
                    freqIdx = unknownFreqIdx
                    unknownFreqCount += 1
                                
                wordIndices.append(wordIdx)
                simIndices.append(simIdx)
                freqIndices.append(freqIdx)

        for word in sentence[1].split(): #supersense text, space separated
                
                
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()] 
                else:
                    wordIdx = unknownIdx
                    unknownSCount += 1
                                
                senseIndices.append(wordIdx)        
        
        #Get the label and map to int
        labelIdx = label2Idx[sentence[2]]
            
        xMatrix.append(wordIndices)
        simMatrix.append(simIndices)
        #print simIndices
        freqMatrix.append(freqIndices)
        #print freqIndices
        sMatrix.append(senseIndices)
        yVector.append(labelIdx)
    
    xMatrix = sequence.pad_sequences(xMatrix, maxlen=maxleng)
    simMatrix = sequence.pad_sequences(simMatrix, maxlen=maxleng)
    freqMatrix = sequence.pad_sequences(freqMatrix, maxlen=maxleng)
    sMatrix = sequence.pad_sequences(sMatrix, maxlen=maxleng2)
    print "Unknowns: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    print "Unknown sims: %.2f%%" % (unknownSimCount/(float(wordCount))*100)
    print "Unknown freq: %.2f%%" % (unknownFreqCount/(float(wordCount))*100)
    print "Unknowns Super: %.2f%%" % (unknownSCount/(float(wordCount))*100)
    return (np.asarray(xMatrix, dtype='int32'), np.asarray(simMatrix, dtype='int32'), np.asarray(freqMatrix, dtype='int32'), np.asarray(sMatrix, dtype='int32'), np.asarray(yVector, dtype='int32'))


   
    
