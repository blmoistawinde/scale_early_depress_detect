import os
from glob import glob
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from data import HierDataModule
from model import HierClassifier
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    print(args.model_dir+"/checkpoints/*.ckpt")
    ckpt_path = glob(args.model_dir+"/checkpoints/*.ckpt")[0]
    model = HierClassifier.load_from_checkpoint(ckpt_path)
    model_type = model.model_type
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    data_module = HierDataModule(args.bs, args.input_dir, tokenizer, args.max_len)
    
    trainer = pl.Trainer(gpus=1, deterministic=True, log_every_n_steps=100)

    results = trainer.test(model=model, datamodule=data_module)
    print(results)


if __name__ == "__main__":
    seed_everything(2021, workers=True)
    parser = argparse.ArgumentParser()
    temp_args, _ = parser.parse_known_args()
    parser = HierClassifier.add_model_specific_args(parser)
    parser.add_argument("--model_dir", type=str, default="./lightning_logs/version_0")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--input_dir", type=str, default="./processed/combined_sim16")
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()
    
    main(args)