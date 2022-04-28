import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from data import FlatDataModule
from model import Classifier
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main(args):
    model = model_type(**vars(args))
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    data_module = FlatDataModule(args.bs, args.input_dir, tokenizer, args.max_len)
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=args.patience,
        mode="max"
    )
    
    trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback], val_check_interval=1.0, max_epochs=100, min_epochs=1, accumulate_grad_batches=args.accumulation, gradient_clip_val=args.gradient_clip_val, deterministic=True, log_every_n_steps=100)

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


if __name__ == "__main__":
    seed_everything(2021, workers=True)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_type", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    temp_args, _ = parser.parse_known_args()
    model_type = Classifier
    parser = model_type.add_model_specific_args(parser)
    parser.add_argument("--bs", type=int, default=6)
    parser.add_argument("--input_dir", type=str, default="./processed/kmeans4")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    parser.add_argument("--find_lr", action="store_true")
    args = parser.parse_args()
    
    main(args)