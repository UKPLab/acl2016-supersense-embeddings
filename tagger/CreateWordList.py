"""
Reads in our train and test files and outputs all words to a vocabulary file

@author: Nils Reimers
"""

filenames = ['data/ritter-eval.tsv', 'data/ritter-dev.tsv', 'data/ritter-train.tsv','data/SEMtrain1.tsv','data/SEMdev1.tsv','data/SEMtest1.tsv']
outputfile = 'lex.txt'

words = set()

for filename in filenames:
    for line in open(filename): 
        line = line.strip()
        
        if len(line) == 0: # or line[0] == '#':
            continue
        
        splits = line.split('\t')
        for word in splits:
           words.add(word)
        #for word in splits[1].split():
           #words.add(word)
fOut = open(outputfile, 'w')    
for word in sorted(words):
    fOut.write(word+'\n')
    
print "Done, words exported"
