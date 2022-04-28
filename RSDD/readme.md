First, `preprocess.py` produce the preprocessed files.

Next, use `sel_posts.py` to produce datasets with screened posts.

Then `bash runme_combine64.sh` can run the experiments of `HAN-BERT (Psych/Full)`.

Other files:
- data.py : defines the datasets and data module
- model.py : defines the models, including BERT and the proposed HAN-BERT (BERTHierClassifierTransAbs)
- main_hier_clf.py : run experiments with HAN-BERT models
- test.py : test with trained checkpoints