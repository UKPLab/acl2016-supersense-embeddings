# -*- coding: utf-8 -*-

"""
createNumpyArrayWithCasing returns the matrices for the word embeddings as well as for the case information
and the Y-vector with the labels
@author: Nils Reimers, Lucie Flekova
"""
import numpy as np
import re
from unidecode import unidecode

def readFile(filepath):
    sentences = []
    sentence = []

    for line in open(filepath):
        line = line.strip()

        if len(line) == 0 or line[0] == '#':
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split('\t')
        if (len(splits)>2): sentence.append([splits[0], splits[2], splits[1]]) #splits1 is pos tag
        else: print len(splits), line

    return sentences


def createNumpyArrayWithCasing(sentences, windowsize, word2Idx, sim2Idx, freq2Idx, label2Idx, caseLookup, posLookup):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']    
    unknownSimIdx = sim2Idx['UNKNOWN']
    paddingSimIdx = sim2Idx['PADDING'] 
    unknownFreqIdx = freq2Idx['UNKNOWN']
    paddingFreqIdx = freq2Idx['PADDING'] 
    
    xMatrix = []
    simMatrix = []
    freqMatrix = []
    caseMatrix = []
    posMatrix = []
    yVector = []
    
    wordCount = 0
    unknownWordCount = 0
    unknownSimCount = 0
    unknownFreqCount = 0
     
    for sentence in sentences:
        targetWordIdx = 0
        
        for targetWordIdx in xrange(len(sentence)):
            
            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = [] 
            simIndices = []    
            caseIndices = []
            posIndices = []
            freqIndices = []

            for wordPosition in xrange(targetWordIdx-windowsize, targetWordIdx+windowsize+1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    simIndices.append(paddingSimIdx)
                    freqIndices.append(paddingFreqIdx)
                    caseIndices.append(caseLookup['PADDING'])
                    posIndices.append(posLookup['PADDING'])
                    continue
                
                word = sentence[wordPosition][0]
                pos = sentence[wordPosition][2]
                #print "POS: ",pos,posLookup[pos]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()] 
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)] 
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1

                if word in sim2Idx:
                    simIdx = sim2Idx[word]
                elif word.lower() in sim2Idx:
                    simIdx = sim2Idx[word.lower()] 
                elif normalizeWord(word) in sim2Idx:
                    simIdx = sim2Idx[normalizeWord(word)] 
                else:
                    simIdx = unknownSimIdx
                    unknownSimCount += 1

                if word in freq2Idx:
                    freqIdx = freq2Idx[word]
                elif word.lower() in freq2Idx:
                    freqIdx = freq2Idx[word.lower()] 
                elif normalizeWord(word) in freq2Idx:
                    freqIdx = freq2Idx[normalizeWord(word)] 
                else:
                    freqIdx = unknownFreqIdx
                    unknownFreqCount += 1

                wordIndices.append(wordIdx)
                simIndices.append(simIdx)
                freqIndices.append(freqIdx)
                caseIndices.append(getCasing(word, caseLookup))
                if (posLookup.has_key(pos)):
                   posIndices.append(posLookup[pos])
                else: posIndices.append(posLookup['other'])
                
            #Get the label and map to int
            #print word, targetWordIdx, sentence[targetWordIdx][1] 
            labelIdx = label2Idx[sentence[targetWordIdx][1]]
            
            #Get the casing            
            xMatrix.append(wordIndices)
            simMatrix.append(simIndices)
            freqMatrix.append(freqIndices)
            caseMatrix.append(caseIndices)
            posMatrix.append(posIndices)
            yVector.append(labelIdx)
    
    #print xMatrix, simMatrix, freqMatrix, yVector
    
    print "Unknowns: %.2f%%" % (unknownWordCount/(float(wordCount))*100)
    print "Unknown sims: %.2f%%" % (unknownSimCount/(float(wordCount))*100)
    print "Unknown freq: %.2f%%" % (unknownFreqCount/(float(wordCount))*100)
    return (np.asarray(xMatrix), np.asarray(simMatrix), np.asarray(freqMatrix), np.asarray(caseMatrix), np.asarray(posMatrix), np.asarray(yVector))

def getCasing(word, caseLookup):   
    casing = 'other'
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    if word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
   
    #print casing
    return caseLookup[casing]

def multiple_replacer(key_values):
    #replace_dict = dict(key_values)
    replace_dict = key_values
    replacement_function = lambda match: replace_dict[match.group(0)]
    pattern = re.compile("|".join([re.escape(k) for k, v in key_values.iteritems()]), re.M)
    return lambda string: pattern.sub(replacement_function, string)


def multiple_replace(string, key_values):
    return multiple_replacer(key_values)(string)


def normalizeWord(line):
    line = unicode(line, "utf-8") #Convert to UTF8
    line = line.replace(u"„", u"\"")

    line = line.lower(); #To lower case

    #Replace all special charaters with the ASCII corresponding, but keep Umlaute
    #Requires that the text is in lowercase before
    replacements = dict(((u"ß", "SZ"), (u"ä", "AE"), (u"ü", "UE"), (u"ö", "OE")))
    replacementsInv = dict(zip(replacements.values(),replacements.keys()))
    line = multiple_replace(line, replacements)
    line = unidecode(line)
    line = multiple_replace(line, replacementsInv)

    line = line.lower() #Unidecode might have replace some characters, like € to upper case EUR

    line = re.sub("([0-9][0-9.,]*)", '0', line) #Replace digits by NUMBER        


    return line.strip();

