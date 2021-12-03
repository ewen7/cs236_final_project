from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import *
import utils as ut
from pprint import pprint

from models.vae import VAE
from train import train
from classifier import Classifier, Net
from run_vae import fit_vae

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset_type', type=str, default='MNIST', help='dataset to utilize')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--iter_max', type=int, default=20000, metavar='IN',
                        help='number of VAE iterations to train (default: 14)')
    parser.add_argument('--sa', type=int, default=100, metavar='IN',
                        help='number sa iters')
    parser.add_argument('--all_epochs', type=int, default=2, metavar='AN',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('-z', type=float, default=3, metavar='Z',
                        help='latent dimension')
    parser.add_argument('--device', type=str, default='cpu', metavar='DEV',
                        help='device')
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
    dataset_type = args.dataset_type.lower()

    if dataset_type == 'mnist':
        transform=transforms.ToTensor()
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)         
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)    
        subset = list(range(0, len(train_dataset), 200))
        train_subdataset = torch.utils.data.Subset(train_dataset, subset)

        all_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        train_loader = torch.utils.data.DataLoader(train_subdataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    elif dataset_type == 'dogs':
        root_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(root_dir, "dogs_data")
        all_loader, train_loader, test_loader = ut.get_dogs_data(data_dir, 32, args.batch_size, 1000, 4) # values from default project
    else:
        raise Exception("Invalid Dataset")

    ## Initial Classifiers
    full_classifier_path = f'checkpoints/full_classifier_{dataset_type}.pt'
    if not os.path.exists(full_classifier_path):
        epochs = args.epochs
        args.epochs = args.all_epochs
        classifier = Classifier(args, device, dataset_type)
        all_classifier = classifier.train(all_loader, test_loader, "all")
        torch.save(all_classifier.state_dict(), full_classifier_path)
        args.epochs = epochs
    else:
        all_classifier = Net(dataset_type)
        all_classifier.load_state_dict(torch.load(full_classifier_path))

    classifier = Classifier(args, device, dataset_type)
    initial_classifier = classifier.train(train_loader, test_loader, "initial")
    print("initial classifier")
    classifier.test_model(initial_classifier, test_loader)

    ## extended
    subset = list(range(100, len(train_dataset), 300))
    new_dataset = torch.utils.data.Subset(train_dataset, subset)
    updated_dataset = torch.utils.data.ConcatDataset([train_subdataset, new_dataset])
    updated_loader = torch.utils.data.DataLoader(updated_dataset, **train_kwargs)

    classifier = Classifier(args, device, dataset_type)
    new_classifier = classifier.train(updated_loader, test_loader, "new")
    print("new classifier")
    classifier.test_model(new_classifier, test_loader)

    # fig, axs = plt.subplots(10, 20)
    # for i, ax in enumerate(axs.flat):
    #     ax.imshow(query_x[i].view(28, 28).detach())
    #     ax.axis('off')
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()

if __name__ == '__main__':
    main()
