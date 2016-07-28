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

* `embeddings-creator` &mdash; code to build your own supersense embeddings using annotated Wikipedia downloadable below
* `tagger` &mdash; code to train and test a supersense tagger 
* `classification` &mdash; code to classify supersense-annotated documents, e.g. for sentiment analysis

## Data description

The supersensed Wikipedia and the computed embeddings are too large to be uploaded here. You can download the corpora from the following links:

https://public.ukp.informatik.tu-darmstadt.de/wikipedia/supertextwiki.tar.xz
(the version with a supersense appended after a term)

https://public.ukp.informatik.tu-darmstadt.de/wikipedia/supertextwiki.tar.xz
(the version with a term replaced by a supersense)

The original excerpt of English Wikipedia, which was used for this mapping, is available for download here: http://lcl.uniroma1.it/babelfied-wikipedia/

The use of the data (which builds upon the links between BabelNet and WordNet synsets) is protected by a CC-BY-NC license: https://creativecommons.org/licenses/by-nc/4.0/legalcode

### Data formats

