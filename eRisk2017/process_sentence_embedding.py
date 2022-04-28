# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch


# %%
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# %%
def get_input_data(file, path):
    post_num = 0
    dom = xml.dom.minidom.parse(path + "/" + file)
    collection = dom.documentElement
    title = collection.getElementsByTagName('TITLE')
    text = collection.getElementsByTagName('TEXT')
    posts = []
    for i in range(len(title)):
        post = title[i].firstChild.data + ' ' + text[i].firstChild.data
        post = re.sub('\n', ' ', post)
        if len(post) > 0:
            posts.append(post.strip())
            post_num = post_num + 1
    return posts, post_num


# %%
train_posts = []
train_tags = []
train_mappings = []
test_posts = []
test_tags = []
test_mappings = []
for base_path in ["negative_examples_anonymous", "negative_examples_test", "positive_examples_anonymous", "positive_examples_test"]:
    base_path = "dataset/"+base_path
    filenames = sorted(os.listdir(base_path))
    for fname in filenames:
        posts, post_num = get_input_data(fname, base_path)
        if "anonymous" in base_path:
            train_mappings.append(list(range(len(train_posts), len(train_posts)+post_num)))
            train_posts.extend(posts)
            train_tags.append(int("positive" in base_path))
        else:
            test_mappings.append(list(range(len(test_posts), len(test_posts)+post_num)))
            test_posts.extend(posts)
            test_tags.append(int("positive" in base_path))


# %%
train_embs = sbert.encode(train_posts, convert_to_tensor=False)
train_embs.shape


test_embs = sbert.encode(test_posts, convert_to_tensor=False)
test_embs.shape


with open("processed/miniLM_L6_embs.pkl", "wb") as f:
    pickle.dump({
        "train_posts": train_posts,
        "train_mappings": train_mappings,
        "train_labels": train_tags,
        "train_embs": train_embs,
        "test_posts": test_posts,
        "test_mappings": test_mappings,
        "test_labels": test_tags,
        "test_embs": test_embs
    }, f)

"""
Do clustering
"""


# with open("processed/miniLM_L6_embs.pkl", "rb") as f:
#     data = pickle.load(f)

# train_posts = data["train_posts"]
# train_mappings = data["train_mappings"]
# train_tags = data["train_labels"]
# train_embs = data["train_embs"]
# test_posts = data["test_posts"]
# test_mappings = data["test_mappings"]
# test_tags = data["test_labels"]
# test_embs = data["test_embs"]

# from cluster_summary import get_kmeans_centroid_ids, get_cluster_summary

# user_posts1 = [train_posts[i] for i in train_mappings[0]]
# user_embs1 = train_embs[train_mappings[0]]
# summaries1 = get_cluster_summary(user_posts1, user_embs1, K=8)
# print("\n".join(summaries1))

# %% [markdown]
# run clustering and save the results

# for K in [8, 16, 32, 64]:
# for K in [16]:
#     os.makedirs(f"./processed/kmeans{K}", exist_ok=True)
#     os.makedirs(f"./processed/kmeans{K}/train", exist_ok=True)
#     os.makedirs(f"./processed/kmeans{K}/test", exist_ok=True)
#     for id0, members in enumerate(tqdm(train_mappings, desc=f"K={K}, train")):
#         user_posts1 = [train_posts[i] for i in members]
#         user_embs1 = train_embs[members]
#         label1 = train_tags[id0]
#         summaries1 = get_cluster_summary(user_posts1, user_embs1, K=K)
#         with open(f"./processed/kmeans{K}/train/{id0:06}_{label1}.txt", "w") as f:
#             f.write("\n".join(summaries1))
#     for id0, members in enumerate(tqdm(test_mappings, desc=f"K={K}, test")):
#         user_posts1 = [test_posts[i] for i in members]
#         user_embs1 = test_embs[members]
#         label1 = test_tags[id0]
#         summaries1 = get_cluster_summary(user_posts1, user_embs1, K=K)
#         with open(f"./processed/kmeans{K}/test/{id0:06}_{label1}.txt", "w") as f:
#             f.write("\n".join(summaries1))




