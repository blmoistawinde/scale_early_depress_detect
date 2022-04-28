import os
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
# from keras.preprocessing.text import Tokenizer, text_to_word_sequence
# from keras.preprocessing.sequence import pad_sequences

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    
    data_module = HierDataModule(args.bs, args.input_dir, tokenizer, args.max_len, args.max_posts, args.use_pkl, args.pad_post, args.keras_tokenizer)
    if data_module.keras_tokenizer is not None:
        args.vocab_size = len(data_module.keras_tokenizer.word_index)
    else:
        args.vocab_size = tokenizer.vocab_size
    model = model_type(**vars(args))
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=args.patience,
        mode="max"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max"
    )
    
    trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback, checkpoint_callback], val_check_interval=1.0, max_epochs=100, min_epochs=1, accumulate_grad_batches=args.accumulation, gradient_clip_val=args.gradient_clip_val, deterministic=True, log_every_n_steps=100)

    if args.find_lr:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)

        # # Results can be found in
        # print(lr_finder.results)

        # # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print(new_lr)

        # # update hparams of the model
        # model.hparams.lr = new_lr
    else:
        trainer.fit(model, data_module)
        # (1) load the best checkpoint automatically (lightning tracks this for you)
        results = trainer.test(datamodule=data_module)
        print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--hier", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2021)
    temp_args, _ = parser.parse_known_args()
    seed_everything(temp_args.seed, workers=True)
    model_type = HierClassifier
    parser = model_type.add_model_specific_args(parser)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--input_dir", type=str, default="./processed/combined_sim16")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    parser.add_argument("--find_lr", action="store_true")
    parser.add_argument("--max_posts", type=int, default=64)
    parser.add_argument("--use_pkl", action="store_true")
    parser.add_argument("--pad_post", action="store_true")
    parser.add_argument("--keras_tokenizer", default="", type=str)
    args = parser.parse_args()
    
    main(args)