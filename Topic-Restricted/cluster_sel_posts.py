import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import xml.dom.minidom
import string
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from cluster_summary import get_kmeans_centroid_ids, get_cluster_summary

in_dir = "split_filter_All-Mental-Health"
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
sbert.cuda()

with open(f"./{in_dir}/train.pkl", "rb") as f:
    user_posts_train, user_labels_train = pickle.load(f)
user_ids_train = [f"{i:05}" for i in range(len(user_labels_train))]

with open(f"./{in_dir}/val.pkl", "rb") as f:
    user_posts_val, user_labels_val = pickle.load(f)
user_ids_val = [f"{i:05}" for i in range(len(user_labels_val))]

with open(f"./{in_dir}/test.pkl", "rb") as f:
    user_posts_test, user_labels_test = pickle.load(f)
user_ids_test = [f"{i:05}" for i in range(len(user_labels_test))]

topK = 64
group = "kmeans"
os.makedirs(f"{in_dir}/{group}_sim{topK}", exist_ok=True)
os.makedirs(f"{in_dir}/{group}_sim{topK}/train", exist_ok=True)
os.makedirs(f"{in_dir}/{group}_sim{topK}/val", exist_ok=True)
os.makedirs(f"{in_dir}/{group}_sim{topK}/test", exist_ok=True)

# train
for id0, posts, label in tqdm(zip(user_ids_train, user_posts_train, user_labels_train), total=len(user_ids_train)):
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    if len(posts) <= 0:
        continue
    embs = sbert.encode(posts)
    summaries1 = get_cluster_summary(posts, embs, K=topK)
    with open(f"{in_dir}/{group}_sim{topK}/train/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(summaries1))
    
print("validation")
# val
for id0, posts, label in tqdm(zip(user_ids_val, user_posts_val, user_labels_val), total=len(user_ids_val)):
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    if len(posts) <= 0:
        continue
    embs = sbert.encode(posts)
    summaries1 = get_cluster_summary(posts, embs, K=topK)
    with open(f"{in_dir}/{group}_sim{topK}/val/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(summaries1))

# test
for id0, posts, label in tqdm(zip(user_ids_test, user_posts_test, user_labels_test), total=len(user_ids_test)):
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    if len(posts) <= 0:
        continue
    embs = sbert.encode(posts)
    summaries1 = get_cluster_summary(posts, embs, K=topK)
    with open(f"{in_dir}/{group}_sim{topK}/test/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(summaries1))


