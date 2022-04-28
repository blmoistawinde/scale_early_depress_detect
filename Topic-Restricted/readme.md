First, `preprocess_wolohan_filter.py` produce the preprocessed datasets.

Next, use `sel_posts.py` to produce datasets with screened posts.

Then `bash runme_combine64.sh` can run the experiments of `HAN-BERT (Psych/Full)`.

Other files:
- data.py : defines the datasets and data module
- model.py : defines the models, including BERT and the proposed HAN-BERT (BERTHierClassifierTransAbs)
- main_hier_clf.py : run experiments with HAN-BERT models
- filter_subreddits.py : include the subreddits to filter out
- cluster_summary.py: utilities for K-Means clustering (Clus)
- cluster_sel_posts.py : produce the K-Means selected posts