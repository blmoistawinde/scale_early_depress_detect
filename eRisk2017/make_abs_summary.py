import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
import json
import torch
from transformers import pipeline
from tqdm import tqdm

K = 16
os.makedirs(f"processed/kmeans{K}_abs256",exist_ok=True)
os.makedirs(f"processed/kmeans{K}_abs256/test",exist_ok=True)
os.makedirs(f"processed/kmeans{K}_abs256/train",exist_ok=True)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6", device=1)

for split in ["train", "test"]:
    base_dir = f"processed/kmeans{K}/{split}"
    out_dir = f"processed/kmeans{K}_abs256/{split}"
    for fname in tqdm(os.listdir(base_dir)):
        text = open(os.path.join(base_dir, fname), encoding="utf-8").read()
        summary = summarizer(text, truncation=True, max_length=256)[0]["summary_text"].strip()
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fo:
            fo.write(summary)