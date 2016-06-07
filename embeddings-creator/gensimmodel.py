import logging, sys, pprint, gensim, nltk, fileinput, os
from nltk.tokenize import word_tokenize
from gensim import utils
import glob, string, re
import random

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


sentences = []

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


for dir in sys.argv[1:]:
        print listdir_fullpath(dir)
        try:
	   f =  fileinput.input(listdir_fullpath(dir),openhook=fileinput.hook_compressed) 
	   for line in f:
                line = line.translate(string.maketrans("",""), string.punctuation).lower()
                sentences.append(line.rstrip('\n').split())                
	   f.close()
        except: 
           pass

model = gensim.models.Word2Vec(sentences, size=300, window=2, min_count=200, sg=1, workers=5, iter=10)
#print model.vocab.keys()
#print len(sentences)

#model.accuracy('questions-words.txt')
#print 'King - man + woman: ', model.most_similar(positive=['woman','king'],negative=['man'],topn=10)

model.save_word2vec_format('super-text-supertextwiki-ALL-200-2-sg.txt')

