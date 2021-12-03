# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
import utils as ut
from models.vae import VAE
from train import train
from pprint import pprint
from torchvision import datasets, transforms

def fit_vae(args, train_loader, model_name, overwrite=True, dataset_type='mnist'):
    vae = VAE(z_dim=args.z, name=model_name, dataset_type=dataset_type).to(args.device)

    writer = ut.prepare_writer(model_name, overwrite_existing=overwrite)
    vae = train(model=vae,
          train_loader=train_loader,
          labeled_subset=None,
          device=args.device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=10000)
    # ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=args.train == 2)

    return vae
