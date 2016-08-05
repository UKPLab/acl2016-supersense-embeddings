# Supersense Embeddings: A Unified Model for Supersense Interpretation, Prediction and Utilization

(Work in progress)

Source code, data, and supplementary materials for our ACL 2016 article. Please use the following citation:

```
@inproceedings{Flekova.Gurevych.2016.ACL,
	author = {Lucie Flekova and Iryna Gurevych},
	title = {Supersense Embeddings: A Unified Model for Supersense Interpretation,
Prediction, and Utilization},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational
               Linguistics (ACL 2016)},
  volume    = {Volume 1: Long Papers},
  year      = {2016},
  address   = {Berlin, Germany},
  pages     = {(to appear)},
  publisher = {Association for Computational Linguistics},
  url       = {https://www.ukp.tu-darmstadt.de/publications/details/?tx_bibtex_pi1[pub_id]=TUD-CS-2016-0104}
}
```
> **Abstract:** Coarse-grained semantic categories such as supersenses have proven useful for a range of downstream tasks such as question answering or machine translation. To date, no effort has been put into integrating the supersenses into distributional word representations. We present a novel joint embedding model of words and supersenses, providing insights into the relationship between words and supersenses in the same vector space. Using these embeddings in a deep neural network model, we demonstrate that the supersense enrichment leads to a significant improvement in a range of downstream classification tasks.

* **Contact person:** Lucie Flekova, flekova@ukp.informatik.tu-darmstadt.de
    * UKP Lab: http://www.ukp.tu-darmstadt.de/
    * TU Darmstadt: http://www.tu-darmstadt.de/

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Project structure

* `embeddings-creator` &mdash; code to build your own supersense embeddings using annotated Wikipedia below
* `tagger` &mdash; code to train and test a supersense tagger 
* `classification` &mdash; code to classify supersense-annotated documents, e.g. for sentiment analysis

## Data description

The supersensed Wikipedia and the computed embeddings are too large to be uploaded here. You can download the corpora from the following links:

https://public.ukp.informatik.tu-darmstadt.de/wikipedia/supertextwiki.tar.xz
(the version with a supersense appended after a term)

https://public.ukp.informatik.tu-darmstadt.de/wikipedia/superwiki.tar.xz
(the version with a term replaced by a supersense)

The source of this data is the Babelfied Wikipedia published here:
http://lcl.uniroma1.it/babelfied-wikipedia/
and described in this paper:
Federico Scozzafava, Alessandro Raganato, Andrea Moro, and Roberto Navigli.
Automatic Identification and Disambiguation of Concepts and Named Entities in the Multilingual Wikipedia. [paper]
Proc. of the 14th Congress of the Italian Association for Artificial Intelligence (AI*IA 2015), Ferrara, Italy, September 23-25th, 2015.

This resource, as well as the original Babelfied Wikipedia, is distributed under the CC-BY-NC licence: http://creativecommons.org/licenses/by-nc-sa/3.0/ The authors of the original Babelfied Wikipedia have no responsibility for the content of this resource.

The pretrained Wikipedia embeddings (words and supersenses together) in W2V format can be downloaded here:
https://public.ukp.informatik.tu-darmstadt.de/wikipedia/supersense-embeddings.txt
(skip-gram, 300 dimensions, window size = 2, min. frequency = 200. Retrain on your own if you wish differently - parameter suitability varies, depending on your target task). Versions with only word_supersense and only supersense also available.

### Data formats


We used the BabelNet API to map the annotations to WordNet 3.0 synsets and from there to their supersenses where available. The resulting files are, despite their name, gzipped plain text files (to facilitate word2vec training) with some nouns and verbs having their supersenses appended. Example:

Annotated file (in this resource):

```
Journalist_noun.person reporting_verb.communication and evaluation_noun.act of video_noun.communication games_noun.act in periodicals_noun.communication began_verb.change from the late_adj.all 1970s to 1980 in general_adj.all coin-operated_adj.pert industry_noun.act magazines_noun.communication like Play_noun.communication Meter_noun.quantity and RePlay, home_noun.location entertainment_noun.act magazines_noun.communication like "Video", as well as magazines_noun.communication focused_verb.change on computing_verb.cognition and new_adj.all information_noun.cognition technologies_noun.cognition like InfoWorld or Popular Electronics.
```
Original file (downloadable separately, article IDs match):

```
Journalist reporting and evaluation of video games in periodicals began from the late 1970s to 1980 in general coin-operated industry magazines like Play Meter and RePlay, home entertainment magazines like "Video", as well as magazines focused on computing and new information technologies like InfoWorld or Popular Electronics.
```

## Experiments

### Requirements

* Tested on: 
* 64-bit Linux versions
* Python 2.7.6
* Theano 0.9.0dev2.dev-RELEASE
    * GPU is recommended but not required
* Keras X.X

### Creating the embeddings

* code to be found in the folder `embeddings-creator`
* download the annotated Wikipedia articles above, unpack the folders and use the script  `runthis.sh`
* to modify training parameters, edit the  `gensimmodel.py` - details are at https://radimrehurek.com/gensim/models/word2vec.html

### Supersense tagger

* code to be found in the folder `tagger`
* using the corpora/embeddings, output the matrix of supersense frequencies and supersense similarities
* modify data paths in  `SupersenseTagger.py`
* You should have CUDA installed on your machine for GPU-enabled computation
    * Refer to http://deeplearning.net/software/theano/install.html
    * This might get sometimes a bit tricky to install
* Run with
```bash
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python SupersenseTagger.py 
```
Output should look like
    
```
Using gpu device 0: Tesla K40c 
Supersenses with Keras with float32
Read in the vocab
Read in data and create matrices
Unknown words train: 21.79%
Unknown words test: 22.34%
units, epochs, batch:  300 70 100
(10559, 'train samples')
(2273, 'test samples')
70 epochs
105 mini batches
6.04 sec for training
2273/2273 [==============================] - 0s     
2273
1 epoch: prec, rec, F1 on dev: 0.212121 0.378319 0.271829, prec, rec, F1 on test: 0.212651 0.355180 0.266028
...
2273/2273 [==============================] - 0s     
2273
70 epoch: prec, rec, F1 on dev: 0.630854 0.613728 0.622173, prec, rec, F1 on test: 0.588156 0.586345 0.587249
```
### Text classification experiments

* code to be found in the folder `classification`
* run with:
```bash
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python TextClassesKeras_CNN_sups.py 'path-to-train-file-pos' 'path-to-train-file-neg' 'path-to-dev-file-pos' 'path-to-dev-file-neg' 'path-to-test-file-pos' 'path-to-test-file-neg'
```
* the train/test files are expected in the two-column format (original text, supersense-replaced text) separated by tab:
```
i 'm sorry to say that this should seal the deal - arnold be not , nor will he be , back .	i verbstative sorry to verbcommunication that this should verbchange the nounpossession - nounperson verbstative not , nor will he verbstative , back .
```
