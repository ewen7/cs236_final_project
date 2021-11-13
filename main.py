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

    transform=transforms.ToTensor()
    # transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])

    # TODO: update this to be subset of dataset
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)         
    subset = list(range(0, len(train_dataset), 200))
    train_subdataset = torch.utils.data.Subset(train_dataset, subset)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    all_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    train_loader = torch.utils.data.DataLoader(train_subdataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    vae = fit_vae(args, train_loader, "vae")
    ## Initial Classifiers
    full_classifier_path = 'checkpoints/full_classifier.pt'
    if not os.path.exists(full_classifier_path):
        epochs = args.epochs
        args.epochs = args.all_epochs
        classifier = Classifier(args, device)
        all_classifier = classifier.train(all_loader, test_loader, "all")
        torch.save(all_classifier.state_dict(), full_classifier_path)
        args.epochs = epochs
    else:
        all_classifier = Net()
        all_classifier.load_state_dict(torch.load(full_classifier_path))

    classifier = Classifier(args, device)
    initial_classifier = classifier.train(train_loader, test_loader, "initial")
    classifier.test_model(initial_classifier, test_loader)

    ## VAE

    mc_samp = 10

    prior_m = torch.zeros(200, args.z)
    prior_v = torch.ones(200, args.z)
    query_z = torch.normal(prior_m, prior_v).requires_grad_(True)

    def simulated_annealing(f, x0, eps=0.1, horizon=100, T=10., cooling=0.1):
        x = torch.tensor(x0)
        x_best = torch.tensor(x0)
        f_cur = f(x)
        gamma = cooling ** (1 / horizon)
        for i in trange(horizon):
            x_prop = x + torch.normal(eps * torch.ones(x.shape))
            f_prop = f(x_prop)
            accept_prob = torch.exp((f_cur - f_prop) / T)
            accepted = torch.rand([x.size(0)], device=device) < accept_prob
            x[accepted] = x_prop[accepted]
            f_cur[accepted] = f_prop[accepted]
            T *= gamma
        return x

    def active_objective(query_z):
        p_z = ut.log_normal(query_z, prior_m, prior_v).exp()
        rec = torch.stack([vae.sample_x_given(query_z) for _ in range(mc_samp)])
        pred_y = initial_classifier(rec.reshape(-1, 1, 28, 28))
        entropies = td.Categorical(logits=pred_y).entropy().reshape(mc_samp, 200).mean(dim=0)
        return p_z * entropies

    query_z = simulated_annealing(active_objective, query_z, horizon=args.sa)
    query_x = vae.sample_x_given(query_z).detach()

    x_labels = all_classifier(query_x.view(query_x.size(0), 1, 28, 28)).argmax(-1)
    new_dataset = torch.utils.data.TensorDataset(query_x.view(query_x.size(0), 1, 28, 28), x_labels)
    updated_dataset = torch.utils.data.ConcatDataset([train_subdataset, new_dataset])
    updated_loader = torch.utils.data.DataLoader(updated_dataset, **train_kwargs)

    classifier = Classifier(args, device)
    new_classifier = classifier.train(updated_loader, test_loader, "new")
    # classifier.test_model(new_classifier, test_loader)

    fig, axs = plt.subplots(10, 20)
    for i, ax in enumerate(axs.flat):
        ax.imshow(query_x[i].view(28, 28).detach())
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

if __name__ == '__main__':
    main()
