First, `process_sentence_embedding.py` embeds all the sentences. 

Next, use `make_dataset.ipynb` to produce the processed datasets.

Then `bash runme_combine16.sh` can run the experiments of `HAN-BERT (Psych/Full)`.

Given pretrained checkpoints on RSDD or Topic-Restricted, we can run `cross_domain_RSDD.ipynb` or `cross_domain_topic_restricted.ipynb` to conduct the cross-domain experiments.

The early detection experiments and the case studies are in `erisk_infer_analyze.ipynb`.

Other files:
- data.py : defines the datasets and data module
- model.py : defines the models, including HAN-GRU, BERT and the proposed HAN-BERT (BERTHierClassifierTransAbs)
- main_hier_clf.py : run experiments with HAN-BERT models
- main_post_clf.py : run experiments with (Flat) BERT models
- ERDE.py : utilities for the calculation of ERDE
- cluster_summary.py: utilities for K-Means clustering (Clus)
- make_abs_summary.py: produce abstractive summarization for (Clus+Abs)
- LIWC_ana.ipynb: lexical analysis with LIWC