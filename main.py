from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import utils as ut
from pprint import pprint

from models.vae import VAE
from train import train
from classifier import Classifier
from run_vae import fit_vae

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # TODO: update this to be subset of dataset
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    subset = list(range(0, len(train_dataset), 200))
    train_subdataset = torch.utils.data.Subset(train_dataset, subset)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    all_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    train_loader = torch.utils.data.DataLoader(train_subdataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    ## Initial Classifiers
    classifier = Classifier(args, device)
    all_classifier = classifier.train(all_loader, test_loader, "all")

    classifier = Classifier(args, device)
    initial_classifier = classifier.train(train_loader, test_loader, "initial")

    ## VAE
    vae = fit_vae(args, train_loader, "vae")

    prior_m = torch.zeros(200, args.z)
    prior_v = torch.ones(200, args.z)
    query_z = torch.normal(prior_m, prior_v).requires_grad_(True)

    def active_objective():
        log_p_z = ut.log_normal(query_z, prior_m, prior_v)
        H


    # expanded_dataset = None
    # expanded_loader = torch.utils.data.DataLoader(expanded_dataset,**train_kwargs)
    # expanded_classifier = classifier.train(expanded_dataset, test_loader, "expanded")

if __name__ == '__main__':
    main()
