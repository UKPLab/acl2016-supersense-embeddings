# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T


import time
import gzip

import SupersenseDataReader
import BIOF1Validation

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Merge, Dropout
from keras.optimizers import SGD, adadelta, RMSprop, adam, adagrad
from keras.utils import np_utils
from keras.layers.embeddings import Embedding

#from KerasLayer.FixedEmbedding import FixedEmbedding


windowSize = 2 # 2 to the left, 2 to the right
#numHiddenUnits = 155
numHiddenUnits = 300

trainFile = 'data/SEMtrain1.tsv'
devFile = 'data/SEMdev1.tsv'
testFile = 'data/SEMtest1.tsv'
#trainFile = 'data/ritter-train.tsv'
#trainFile = 'data/joined-train.tsv'
#devFile = 'data/ritter-dev.tsv'
#testFile = 'data/ritter-eval.tsv'
#testFile = 'data/in-house-eval.tsv'

print "Supersenses with Keras with %s" % theano.config.floatX


#####################
#
# Read in the vocab
#
#####################
print "Read in the vocab"
#vocabPath =  'embeddings/GoogleNews-vectors-negative300.vocab_sub' #'embeddings/GoogleVecs.txt_subb' 
#vocabPath = 'embeddings/super-text-supertext-wiki-sub250-150-2-sg.txt_sub' #'semcor-embed.sub'
vocabPath= '/home/local/UKP/flekova/superwiki-ALL-200-2-sg.txt'
similarityPath =   'embeddings/similarities-nosuper.txt_sub' 
#frequencyPath = '/home/local/UKP/flekova/supertextwikiFreq.txt'
frequencyPath =   'embeddings/wikipediaSenseFrequencies.txt_sub' 


word2Idx = {} #Maps a word to the index in the embeddings matrix
sim2Idx = {} #Maps a word to the index in the embeddings matrix
freq2Idx = {}
embeddings = [] #Embeddings matrix
similarities = [] #Similairty embeddings matrix
freqs = []

with open(vocabPath, 'r') as fIn:
    idx = 0               
    for line in fIn:
        split = line.strip().split(' ')                
        embeddings.append(np.array([float(num) for num in split[1:]]))
        word2Idx[split[0]] = idx
        idx += 1

with open(similarityPath, 'r') as fIn2:
    idx = 0               
    for linne in fIn2:
        splitt = linne.strip().split(' ')                
        similarities.append(np.array([float(num) for num in splitt[1:]]))
        sim2Idx[splitt[0]] = idx
        idx += 1

with open(frequencyPath, 'r') as fIn3:
    idx = 0               
    for linne in fIn3:
        splitt = linne.strip().split(' ')                
        freqs.append(np.array([float(num) for num in splitt[1:]]))
        freq2Idx[splitt[0]] = idx
        idx += 1
        
embeddings = np.asarray(embeddings, dtype=theano.config.floatX)
similarities = np.asarray(similarities, dtype=theano.config.floatX)
freqs = np.asarray(freqs, dtype=theano.config.floatX)

embedding_size = embeddings.shape[1]
similarities_size = similarities.shape[1]
freqs_size = freqs.shape[1]

        
# Create a mapping for our labels
label2Idx = {'O':0,'0':0}
#label2Idx = {'0':0}
idx = 1

for bioTag in ['B-', 'I-']:
    for nerClass in ['verb.cognition', 'verb.change', 'verb.body', 'verb.communication','verb.competition','verb.consumption',
'verb.contact','verb.creation','verb.emotion','verb.motion','verb.perception','verb.possession','verb.social','verb.stative','verb.weather',
'noun.act','noun.animal','noun.artifact','noun.attribute','noun.body','noun.cognition','noun.communication','noun.event','noun.feeling','noun.food',
'noun.group','noun.location','noun.motive','noun.object','noun.person','noun.phenomenon','noun.plant','noun.possession','noun.process',
'noun.quantity','noun.relation','noun.shape','noun.state','noun.substance','noun.time','noun.Tops']:
        #for subtype in ['', 'deriv', 'part']:
            label2Idx[bioTag+nerClass] = idx 
            idx += 1
            
#Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}


#Casing matrix
caseLookup = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'PADDING':5}

#POS matrix
posLookup = {'.':0, 'ADJ':1, 'ADP':2, 'ADV':3, 'CONJ':4, 'DET':5, 'NOUN':6, 'NUM':7, 'PRON':8, 'PRT':9, 'VERB':10, 'X':11,
'\'\'':0,'(':0,')':0,',':0,'.':0,':':0,'CC':4,'CD':7,'DT':5,'EX':5,'FW':11,'HT':11,'IN':2,'JJ':1,'JJR':1,
'JJS':1,'LS':11,'MD':10,'NN':6,'NNP':6,'NNPS':6,'NNS':6,'NONE':11,'O':11,'PDT':5,'POS':9,'PRP':8,'PRP$':8,
'RB':3,'RBR':3,'RBS':3,'RP':9,'RT':0,'SYM':11,'TD':11,'TO':9,'UH':11,'URL':11,'USR':6,'VB':10,'VBD':10,
'VBG':10,'VBN':10,'VBP':10,'VBZ':10,'VPP':10,'WDT':5,'WH':11,'WP':8,'WRB':3,'PADDING':49,'other':50,  '$':50,'``':50,'WP':50,'WP$':50} #,'$':51,'WP':52}

#posLookup = {'\'\'':0,'(':1,')':2,',':3,'.':4,':':5,'CC':6,'CD':7,'DT':8,'EX':9,'FW':10,'HT':11,'IN':12,'JJ':13,'JJR':14,
#'JJS':15,'LS':16,'MD':17,'NN':18,'NNP':19,'NNPS':20,'NNS':21,'NONE':22,'O':23,'PDT':24,'POS':25,'PRP':26,'PRP$':27,
#'RB':28,'RBR':29,'RBS':30,'RP':31,'RT':32,'SYM':33,'TD':34,'TO':35,'UH':36,'URL':37,'USR':38,'VB':39,'VBD':40,
#'VBG':41,'VBN':42,'VBP':43,'VBZ':44,'VPP':45,'WDT':46,'WP':47,'WRB':48,'PADDING':49,'other':50, '$':50,'``':50,'WP':50}

superLookup = {'verb.cognition':1, 'verb.change':2, 'verb.body':3, 'verb.communication':4,'verb.competition':5,'verb.consumption':6,
'verb.contact':7,'verb.creation':8,'verb.emotion':9,'verb.motion':10,'verb.perception':11,'verb.possession':12,'verb.social':13,
'verb.stative':14,'verb.weather':15,'noun.act':16,'noun.animal':17,'noun.artifact':18,'noun.attribute':19,'noun.body':20,'noun.cognition':21,
'noun.communication':22,'noun.event':23,'noun.feeling':24,'noun.food':25,'noun.group':26,'noun.location':27,'noun.motive':28,
'noun.object':29,'noun.person':30,'noun.phenomenon':31,'noun.plant':32,'noun.possession':33,'noun.process':34,
'noun.quantity':35,'noun.relation':36,'noun.shape':37,'noun.state':38,'noun.substance':39,'noun.time':40,'noun.Tops':41,'PADDING':42}
            
caseMatrix = np.identity(len(caseLookup), dtype=theano.config.floatX)
     
posMatrix = np.identity(len(posLookup), dtype=theano.config.floatX)

# Read in data   
print "Read in data and create matrices"    
train_sentences = SupersenseDataReader.readFile(trainFile)
dev_sentences = SupersenseDataReader.readFile(devFile)
test_sentences = SupersenseDataReader.readFile(testFile)

# Create numpy arrays
train_x, train_sim_x, train_freq_x, train_case_x, train_pos_x, train_y = SupersenseDataReader.createNumpyArrayWithCasing(train_sentences, windowSize, word2Idx, sim2Idx, freq2Idx, label2Idx, caseLookup, posLookup)
dev_x, dev_sim_x, dev_freq_x, dev_case_x, dev_pos_x, dev_y = SupersenseDataReader.createNumpyArrayWithCasing(dev_sentences, windowSize, word2Idx, sim2Idx, freq2Idx, label2Idx, caseLookup, posLookup)
test_x, test_sim_x, test_freq_x, test_case_x, test_pos_x, test_y = SupersenseDataReader.createNumpyArrayWithCasing(test_sentences, windowSize, word2Idx, sim2Idx, freq2Idx, label2Idx, caseLookup, posLookup)


#####################################
#
# Create the Keras Network
#
#####################################


# Create the train and predict_labels function
n_in = 2*windowSize+1
n_hidden = numHiddenUnits
n_out = len(label2Idx)

number_of_epochs = 50
minibatch_size = 100
embedding_size = embeddings.shape[1]

print "units, epochs, batch: ", n_hidden, number_of_epochs, minibatch_size

#dim_case = 6

x = T.imatrix('x')  # the data, one word+context per row
y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
        
        
print "Embeddings shape sim ",similarities.shape        
print "Embeddings shape words ",embeddings.shape

words = Sequential()
words.add(Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in,  weights=[embeddings]))        
words.add(Dropout(0.25))
words.add(Flatten())


senses = Sequential()
senses.add(Embedding(output_dim=similarities.shape[1], input_dim=similarities.shape[0], input_length=n_in,  weights=[similarities]))       
senses.add(Dropout(0.25))
senses.add(Flatten())

freq = Sequential()
freq.add(Embedding(output_dim=freqs.shape[1], input_dim=freqs.shape[0], input_length=n_in,  weights=[freqs]))       
freq.add(Dropout(0.25))
freq.add(Flatten())

casing = Sequential()
casing.add(Embedding(output_dim=caseMatrix.shape[1], input_dim=caseMatrix.shape[0], input_length=n_in, weights=[caseMatrix]))       
casing.add(Flatten())

pos = Sequential()
pos.add(Embedding(output_dim=posMatrix.shape[1], input_dim=len(posLookup), input_length=n_in, weights=[posMatrix]))       
pos.add(Flatten())

temp3 = Sequential()
temp3.add(Merge([senses, pos], mode='concat'))

temp = Sequential()
temp.add(Merge([temp3, words], mode='concat'))

temp2 = Sequential()
temp2.add(Merge([temp, casing], mode='concat'))

model = Sequential()
model.add(Merge([temp2, freq], mode='concat'))

#relu = keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
model.add(Dense(output_dim=n_hidden, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(output_dim=50, input_dim=n_hidden, activation='relu'))
model.add(Dense(output_dim=n_out, init='glorot_uniform', activation='softmax'))
            
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#model.compile(loss='mean_squared_error', optimizer=sgd)

print(train_x.shape[0], 'train samples')
print(train_x.shape[1], 'train dimension')
print(test_x.shape[0], 'test samples')

train_y_cat = np_utils.to_categorical(train_y, n_out)
 
#Function that helps to iterate over our data in minibatches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
        
print "%d epochs" % number_of_epochs
print "%d mini batches" % (len(train_x)/minibatch_size)

for epoch in xrange(number_of_epochs):    
    start_time = time.time()
    
    model.fit([train_sim_x, train_pos_x, train_x, train_case_x, train_freq_x], train_y_cat, nb_epoch=1, batch_size=minibatch_size, verbose=0, shuffle=False)
    #for batch in iterate_minibatches(train_x, train_y_cat, minibatch_size, shuffle=False):
    #    inputs, targets = batch
    #    model.train_on_batch(inputs, targets)   
        

    print "%.2f sec for training" % (time.time() - start_time)
  
    pre_dev, rec_dev, f1_dev = BIOF1Validation.compute_f1(model.predict_classes([dev_sim_x, dev_pos_x, dev_x, dev_case_x, dev_freq_x], verbose=0), dev_y, idx2Label)
    pre_test, rec_test, f1_test = BIOF1Validation.compute_f1(model.predict_classes([test_sim_x, test_pos_x, test_x, test_case_x, test_freq_x], verbose=1), test_y, idx2Label)
    print test_y.shape[0]
    print "%d epoch: prec, rec, F1 on dev: %f %f %f, prec, rec, F1 on test: %f %f %f" % (epoch+1, pre_dev, rec_dev, f1_dev, pre_test, rec_test, f1_test)
    #if epoch==stop_epoch:
       #for i in range(0, test_y.shape[0]):
          #print i, idx2Label[model.predict_classes([test_x, test_pos_x, test_case_x], verbose=0)[i]], idx2Label[test_y[i]]
    
        
