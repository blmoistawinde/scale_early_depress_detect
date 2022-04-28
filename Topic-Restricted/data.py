from collections import defaultdict
import os
import re
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from scipy.io import loadmat
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from collections import defaultdict
from tqdm import tqdm

class FlatDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", max_post_len=-1):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_post_len = max_post_len
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, split)
        for fname in os.listdir(input_dir2):
            label = float(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"
            sample = {}
            if max_post_len == -1:  # no limit
                sample["text"] = open(os.path.join(input_dir2, fname), encoding="utf-8").read()
            else:
                text0 = ""
                with open(os.path.join(input_dir2, fname), encoding="utf-8") as f:
                    for line in f:
                        text0 += " ".join(line.split()[:max_post_len])
                sample["text"] = text0
            tokenized = tokenizer(sample["text"], truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_flat(data):
    labels = []
    processed_batch = defaultdict(list)
    for item, label in data:
        for k, v in item.items():
            processed_batch[k].append(v)
        labels.append(label)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        processed_batch[k] = torch.LongTensor(processed_batch[k])
    labels = torch.FloatTensor(labels)
    return processed_batch, labels

class FlatDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len, max_post_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_post_len = max_post_len
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "train", self.max_post_len)
            self.val_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "val", self.max_post_len)
        elif stage == "test":
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test", self.max_post_len)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_flat, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, collate_fn=my_collate_flat, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_flat, pin_memory=True, num_workers=4)

class HierDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", max_posts=64, pad_post=False):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, split)
        for fname in tqdm(os.listdir(input_dir2)):
            label = float(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"
            sample = {}
            posts = open(os.path.join(input_dir2, fname), encoding="utf-8").read().strip().split("\n")[:max_posts]
            if pad_post and len(posts) < max_posts:   # pad with empty posts
                posts = posts + ["" for i in range(len(posts), max_posts)]
            tokenized = tokenizer(posts, truncation=True, padding='max_length', max_length=max_len, return_tensors='np')
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

class HierPklDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", max_posts=64, pad_post=True):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.data = []
        self.labels = []
        split2fname = {"train": "train.pkl", "val": "val.pkl", "test": "test.pkl"}
        full_dir = os.path.join(input_dir, split2fname[split])
        with open(full_dir, "rb") as f:
            user_posts, user_labels = pickle.load(f)
        self.labels = user_labels   
        for posts in tqdm(user_posts):
            sample = {}
            posts = posts[-max_posts:]   # take the last posts
            if pad_post and len(posts) < max_posts:   # pad with empty posts
                posts = posts + ["" for i in range(len(posts), max_posts)]
            tokenized = tokenizer(posts, truncation=True, padding='max_length', max_length=max_len, return_tensors='np', return_token_type_ids=False, return_attention_mask=False)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
        del user_posts

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]


def my_collate_hier(data):
    labels = []
    processed_batch = []
    for item, label in data:
        user_feats = {}
        for k, v in item.items():
            user_feats[k] = torch.LongTensor(v)
        processed_batch.append(user_feats)
        labels.append(label)
    labels = torch.FloatTensor(np.array(labels))
    return processed_batch, labels

class HierDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len, max_posts, use_pkl=False, pad_post=False):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.pad_post = pad_post
        self.dataset_type = HierPklDataset if use_pkl else HierDataset
    
    def setup(self, stage):
        if stage == "fit":
            print("Loading train set")
            self.train_set = self.dataset_type(self.input_dir, self.tokenizer, self.max_len, "train", self.max_posts, self.pad_post)
            print("Loading val set")
            self.val_set = self.dataset_type(self.input_dir, self.tokenizer, self.max_len, "val", self.max_posts, self.pad_post)
        elif stage == "test":
            self.test_set = self.dataset_type(self.input_dir, self.tokenizer, self.max_len, "test", self.max_posts, self.pad_post)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_hier, shuffle=True, pin_memory=False, num_workers=min(4, self.bs))

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=min(4, self.bs))
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=min(4, self.bs))



def infer_preprocess(tokenizer, texts, max_len):
    batch = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        if k in batch:
            batch[k] = torch.LongTensor(batch[k])
    return batch
        
