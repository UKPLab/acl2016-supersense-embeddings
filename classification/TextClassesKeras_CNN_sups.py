# -*- coding: utf-8 -*-
"""
CNN-LSTM
"""
import numpy as np
import theano
import theano.tensor as T


import time
import gzip
import sys

import TextClassesReader
#import BIOF1Validation

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU

#from KerasLayer.FixedEmbedding import FixedEmbedding

np.random.seed(1014)

trainPosFile = sys.argv[1]
trainNegFile = sys.argv[2]
testPosFile = sys.argv[3]
testNegFile = sys.argv[4]
devPosFile = sys.argv[5]
devNegFile = sys.argv[6]


#####################
#
# Read in the vocab
#
#####################
print "Read in the vocab"
vocabPath= 'embeddings/superwiki.txt'
similarityPath = 'embeddings/similarities-wiki-sg.txt' 
frequencyPath =   'embeddings/wikipediaSenseFrequencies.txt' 


word2Idx = {} #Maps a word to the index in the embeddings matrix
sim2Idx = {} #embedings with supersense similarities of a word
freq2Idx = {} #embeddings with supersense frequencies of a word
embeddings = [] #Embeddings matrix
similarities = [] #sims to supersenses
freqs = [] #freqs of supersenses

def readthevoc(vocPath, resultArray, wordIdx):
   with open(vocPath, 'r') as fIn:
     idx = 0
     for line in fIn:
        split = line.strip().split(' ')
        resultArray.append(np.array([float(num) for num in split[1:]]))
        wordIdx[split[0]] = idx
        idx += 1
     return resultArray, wordIdx

embeddings, word2Idx = readthevoc(vocabPath, embeddings,word2Idx)
similarities, sim2Idx = readthevoc(similarityPath, similarities,sim2Idx)
freqs, freq2Idx = readthevoc(frequencyPath, freqs,freq2Idx)
        
embeddings = np.asarray(embeddings, dtype='float32')
similarities = np.asarray(similarities, dtype='float32')
freqs = np.asarray(freqs, dtype='float32')


embedding_size = embeddings.shape[1]

# Create a mapping for our labels (useless here, but for general case)
label2Idx = {'0':0, '1':1, 0:0, 1:1} 
            
#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}
            
     
# Read in data   
print "Read in data and create matrices"    
train_pos_sentences, train_pos_maxlen, train_pos_maxlen2 = TextClassesReader.readFile(trainPosFile,1)
dev_pos_sentences, dev_pos_maxlen, dev_pos_maxlen2 = TextClassesReader.readFile(devPosFile,1)
test_pos_sentences, test_pos_maxlen, test_pos_maxlen2 = TextClassesReader.readFile(testPosFile,1)
    
train_neg_sentences, train_neg_maxlen, train_neg_maxlen2 = TextClassesReader.readFile(trainNegFile,0)
dev_neg_sentences, dev_neg_maxlen, dev_neg_maxlen2 = TextClassesReader.readFile(devNegFile,0)
test_neg_sentences, test_neg_maxlen, test_neg_maxlen2 = TextClassesReader.readFile(testNegFile,0)

train_sentences = train_pos_sentences + train_neg_sentences
dev_sentences = dev_pos_sentences + dev_neg_sentences
test_sentences = test_pos_sentences + test_neg_sentences

maxlen=max(train_pos_maxlen, dev_pos_maxlen, test_pos_maxlen, train_neg_maxlen, dev_neg_maxlen, test_neg_maxlen)
maxlen2=max(train_pos_maxlen2, dev_pos_maxlen2, test_pos_maxlen2, train_neg_maxlen2, dev_neg_maxlen2, test_neg_maxlen2)
#maxlen2 = 0
print "MAXLEN: ", maxlen, maxlen2


# Create numpy arrays
train_x, train_sim_x, train_freq_x, train_sense_x, train_y = TextClassesReader.createNumpyArray(train_sentences, maxlen, maxlen2, word2Idx, sim2Idx, freq2Idx, label2Idx)
dev_x, dev_sim_x, dev_freq_x, dev_sense_x, dev_y = TextClassesReader.createNumpyArray(dev_sentences, maxlen, maxlen2, word2Idx, sim2Idx, freq2Idx, label2Idx)
test_x, test_sim_x, test_freq_x, test_sense_x, test_y = TextClassesReader.createNumpyArray(test_sentences, maxlen, maxlen2, word2Idx, sim2Idx, freq2Idx, label2Idx)


#####################################
#
# Create the  Network
#
#####################################

n_out = len(label2Idx)

       
print(len(train_x), 'train sequences')
print(len(test_x), 'test sequences')
print(len(train_sense_x), 'train sense sequences')
print(len(test_sense_x), 'test sense sequences')

print("Pad sequences (samples x time)")
train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)
dev_x = sequence.pad_sequences(dev_x, maxlen=maxlen)
train_sim_x = sequence.pad_sequences(train_sim_x, maxlen=maxlen)
test_sim_x = sequence.pad_sequences(test_sim_x, maxlen=maxlen)
dev_sim_x = sequence.pad_sequences(dev_sim_x, maxlen=maxlen)
train_freq_x = sequence.pad_sequences(train_freq_x, maxlen=maxlen)
test_freq_x = sequence.pad_sequences(test_freq_x, maxlen=maxlen)
dev_freq_x = sequence.pad_sequences(dev_freq_x, maxlen=maxlen)
train_sense_x = sequence.pad_sequences(train_sense_x, maxlen=maxlen2)
test_sense_x = sequence.pad_sequences(test_sense_x, maxlen=maxlen2)
dev_sense_x = sequence.pad_sequences(dev_sense_x, maxlen=maxlen2)
print('X_train shape:', train_x.shape)
print('X_test shape:', test_x.shape)

# Train_y is a 1-dimensional vector containing the index of the label
# With np_utils.to_categorical we map it to a 1 hot matrix
train_y_cat = np_utils.to_categorical(train_y, n_out)
test_y_cat = np_utils.to_categorical(dev_y, n_out)
dev_y_cat = np_utils.to_categorical(test_y, n_out)
        
print "Embeddings shape",embeddings.shape
#print "Sims shape",similarities.shape

print test_y_cat

batch_size = 100
#embedding_dims = 300
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 50

words = Sequential()
# Embeddings layers, lookups the word indices and maps them to their dense vectors. FixedEmbeddings are _not_ updated during training
# If you switch it to an Embedding-Layer, they will be updated (training time increases significantly)   
words.add(Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=maxlen,  weights=[embeddings]))  
words.add(Dropout(0.25))
# we add a Convolution1D, which will learn nb_filter word group filters of size filter_length:
words.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
words.add(MaxPooling1D(pool_length=2))
words.add(LSTM(70))
words.add(Dropout(0.25))

#sims = Sequential()
#sims.add(Embedding(output_dim=similarities.shape[1], input_dim=similarities.shape[0], input_length=maxlen,  weights=[similarities]))       
#sims.add(Dropout(0.25))
#sims.add(Convolution1D(nb_filter=nb_filter,
#                        filter_length=filter_length,
#                        border_mode='valid',
#                        activation='relu',
#                        subsample_length=1))
#sims.add(MaxPooling1D(pool_length=2))
#sims.add(LSTM(20))
#sims.add(Dropout(0.25))

#freq = Sequential()
#freq.add(Embedding(output_dim=freqs.shape[1], input_dim=freqs.shape[0], input_length=maxlen,  weights=[freqs]))       
#freq.add(Dropout(0.25))
#freq.add(Convolution1D(nb_filter=nb_filter,
#                        filter_length=filter_length,
#                        border_mode='valid',
#                        activation='relu',
#                        subsample_length=1))
#freq.add(MaxPooling1D(pool_length=2))
#freq.add(LSTM(20))
#freq.add(Dropout(0.25))

senses = Sequential()  
#senses.add(FixedEmbedding(output_dim=similarities.shape[1], input_dim=similarities.shape[0], input_length=maxlen,  weights=[similarities]))  
senses.add(Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=maxlen,  weights=[embeddings])) 
senses.add(Dropout(0.25))
senses.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
#senses.add(MaxPooling1D(pool_length=2))
senses.add(LSTM(30))
senses.add(Dropout(0.25))
#senses.add(Flatten())

model = Sequential()
model.add(Merge([words, senses], mode='concat'))

#temp2 = Sequential()
#temp2.add(Merge([temp1, freq], mode='concat'))

#model = Sequential()
#model.add(Merge([temp, freq], mode='concat'))

# We add a vanilla hidden layer:

model.add(Dense(60))
model.add(Dropout(0.30))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
#model = Sequential()
#model.add(Merge([words, senses], mode='concat'))
#senses.add(Dense(30))
#senses.add(Dropout(0.30))
#words.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode='binary',metrics=['accuracy'])
model.fit([train_x,train_sense_x], train_y, batch_size=batch_size,
          nb_epoch=5, show_accuracy=True, #validation_split=0.2,
          validation_data=([test_x,test_sense_x],test_y), 
          shuffle=True)
score, acc = model.evaluate([test_x,test_sense_x],test_y, batch_size=batch_size)
#print('Test score:', score)
print('Test accuracy:', acc)

#model.evaluate(test_x,  test_y, show_accuracy=True, verbose=2)
#y_pred = model.predict_classes([test_x,test_sim_x,test_freq_x,test_sense_x], verbose=1)
#for i in range(0,len(y_pred)):
#       print  y_pred[i], test_y[i]
#print "%d epoch:  prec, rec, F1 on test: %f %f %f" % (epoch+1, pre_test, rec_test, f1_test)




    
        
