import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
)

import wandb
from torch.utils.tensorboard import SummaryWriter

from importlib import import_module
import argparse
import os, sys
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

## custom lib
import torch_utils
from torch_dataset import MyDataset
from torch_loss import create_criterion

ROOT_DIR = Path('.').reslove()
sys.path.append(ROOT_DIR.as_posix())

import torch_config

CONFIG = torch_config.NewConfig
print(f"""
*** Current CONFIG: ***
{CONFIG.get_dict()}
""")

def val(csv_path, model_path, args):
    torch_utils.seed_everything(args.seed)

    save_path = model_path / args.name

    # --settings
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    print(f"""
    *** Current Device: ***
    {device}
    """)

    # --dataset
    dataset_module = getattr(import_module('dataset'), args.dataset) # default: MyDataset
    dataset = dataset_module(
        ...
    )
    num_classes = 0

    # --augmentation

    # --data_loader
    train_set, val_set = dataset.split.dataset()
    num_workers = 8

    train_loader = DataLoader(
        train_set, 
        batch_size=args.train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=use_cuda,
        drop_list=True,
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=args.valid_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=use_cuda,
        drop_list=True,
    )

    # --model
    model_module = getattr(import_module('model'), args.model) # default: MyModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = nn.DataParallel(model)

    # --loss & metric
    criterion = nn.MSELoss()
    opt_module = getattr(import_module('torch.optim'), args.optimizer)



