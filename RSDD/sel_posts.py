# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # sel_posts

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


# %%
questionaire_single = [
    "I feel sad.",
    "I am discouraged about my future.",
    "I always fail.",
    "I don't get pleasure from things.",
    "I feel quite guilty.",
    "I expected to be punished.",
    "I am disappointed in myself.",
    "I always criticize myself for my faults.",
    "I have thoughts of killing myself.",
    "I always cry.",
    "I am hard to stay still.",
    "It's hard to get interested in things.",
    "I have trouble making decisions.",
    "I feel worthless.",
    "I don't have energy to do things.",
    "I have changes in my sleeping pattern.",
    "I am always irritable.",
    "I have changes in my appetite.",
    "I feel hard to concentrate on things.",
    "I am too tired to do things.",
    "I have lost my interest in sex."
]


# %%
depression_texts = [
    "I feel depressed.",
    "I am diagnosed with depression.",
    "I am treating my depression."
]


# %%
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# %%
depression_embs = sbert.encode(depression_texts)
questionaire_embs = sbert.encode(questionaire_single)
combined_embs = np.concatenate([depression_embs, questionaire_embs], axis=0)


# %%
with open("./processed/training.pkl", "rb") as f:
    user_ids_train, user_posts_train, user_labels_train = pickle.load(f)

with open("./processed/validation.pkl", "rb") as f:
    user_ids_val, user_posts_val, user_labels_val = pickle.load(f)

with open("./processed/testing.pkl", "rb") as f:
    user_ids_test, user_posts_test, user_labels_test = pickle.load(f)

# %% [markdown]
# ## save dataset

# %%
topK = 64
for group in ["depress", "questionaire", "combined", "last"]:
    os.makedirs(f"processed/{group}_sim{topK}", exist_ok=True)
    os.makedirs(f"processed/{group}_sim{topK}/train", exist_ok=True)
    os.makedirs(f"processed/{group}_sim{topK}/val", exist_ok=True)
    os.makedirs(f"processed/{group}_sim{topK}/test", exist_ok=True)

# train
for id0, posts, label in tqdm(zip(user_ids_train, user_posts_train, user_labels_train), total=len(user_ids_train)):
    with open(f"processed/last_sim{topK}/train/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in posts[-topK:]))
    embs = sbert.encode(posts)
    pair_sim = cosine_similarity(embs, combined_embs)
    # depression only
    sim_scores = pair_sim[:, :3].max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"processed/depress_sim{topK}/train/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
    # questionaire only
    sim_scores = pair_sim[:, 3:].max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"processed/questionaire_sim{topK}/train/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
    # combined
    sim_scores = pair_sim.max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"processed/combined_sim{topK}/train/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

print("validation")
# val
for id0, posts, label in tqdm(zip(user_ids_val, user_posts_val, user_labels_val), total=len(user_ids_val)):
    with open(f"processed/last_sim{topK}/val/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in posts[-topK:]))
    embs = sbert.encode(posts)
    pair_sim = cosine_similarity(embs, combined_embs)
    # depression only
    sim_scores = pair_sim[:, :3].max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"processed/depress_sim{topK}/val/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
    # questionaire only
    sim_scores = pair_sim[:, 3:].max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"processed/questionaire_sim{topK}/val/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
    # combined
    sim_scores = pair_sim.max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"processed/combined_sim{topK}/val/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

# test
for id0, posts, label in tqdm(zip(user_ids_test, user_posts_test, user_labels_test), total=len(user_ids_test)):
    with open(f"processed/last_sim{topK}/test/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in posts[-topK:]))
    embs = sbert.encode(posts)
    pair_sim = cosine_similarity(embs, combined_embs)
    # depression only
    sim_scores = pair_sim[:, :3].max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"processed/depress_sim{topK}/test/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
    # questionaire only
    sim_scores = pair_sim[:, 3:].max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"processed/questionaire_sim{topK}/test/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
    # combined
    sim_scores = pair_sim.max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"processed/combined_sim{topK}/test/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))


# %%



