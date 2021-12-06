from __future__ import print_function
import argparse
import os
import numpy as np
import random
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
from classifier import Classifier, Net # Net
from run_vae import fit_vae
import pdb

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
    parser.add_argument('--start', type=int, default=200, help='start size')
    parser.add_argument('--iter-max', type=int, default=20000, metavar='IN',
                        help='number of VAE iterations to train (default: 14)')
    parser.add_argument('--random', action='store_true', default=False,
                        help='generate random queries from generative model')
    parser.add_argument('--sa', type=int, default=100, metavar='IN',
                        help='number sa iters')
    parser.add_argument('--all_epochs', type=int, default=2, metavar='AN',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--query-size', type=int, default=50,
                        help='query size')
    parser.add_argument('--mc-samp', type=int, default=10,
                        help='MC samples to use')
    parser.add_argument('--horizon', type=int, default=10,
                        help='horizon')
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
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='run baseline')
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

    # TODO: update this to be subset of dataset
    
    if dataset_type == 'mnist':
        transform=transforms.ToTensor()
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)         
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)    
        subset = random.Random(args.seed).sample(list(range(0, len(train_dataset))), args.start)
        train_subdataset = torch.utils.data.Subset(train_dataset, subset)
    elif dataset_type == 'dogs':
        root_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(root_dir, "dogs_data")
        train_dataset, train_subdataset, test_dataset = ut.get_dogs_data(data_dir, 32, args.batch_size, 1000)
        # all_loader, train_loader, test_loader = ut.get_dogs_data(data_dir, 32, args.batch_size, 1000) # values from default project
    else:
        raise Exception("Invalid Dataset")
    
    imsize = 28 if dataset_type == 'mnist' else 32
    all_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    train_loader = torch.utils.data.DataLoader(train_subdataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

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
        all_classifier = Net(dataset_type) #if dataset_type == "mnist" else NetDogs(dataset_type)
        all_classifier.load_state_dict(torch.load(full_classifier_path))

    ## Baseline - only works for MNIST
    if args.baseline:
        current_dataset = train_subdataset
        classifier = Classifier(args, device, dataset_type)
        baseline_classifier = classifier.train(train_loader, test_loader, "baseline")
        results = [classifier.test_model(baseline_classifier, test_loader)]
        for i in range(args.horizon):
            subset = random.Random(args.seed + i).sample(list(range(0, len(train_dataset))), args.query_size)
            new_dataset = torch.utils.data.Subset(train_dataset, subset)
            current_dataset = torch.utils.data.ConcatDataset([current_dataset, new_dataset])
            loader = torch.utils.data.DataLoader(current_dataset, **train_kwargs)
            classifier = Classifier(args, device, dataset_type)
            baseline_classifier = classifier.train(loader, test_loader, "baseline")
            results += [classifier.test_model(baseline_classifier, test_loader)]
        print(results)
        return

    classifier = Classifier(args, device, dataset_type)
    initial_classifier = classifier.train(train_loader, test_loader, "initial")
    results = [classifier.test_model(initial_classifier, test_loader)]

    ## VAE
    vae_path = 'checkpoints/vae_{dataset_type}.pt' if dataset_type=='mnist' else 'checkpoints/vae/model-10000.pt'
    if not os.path.exists(vae_path):
        print("fitting vae...")
        vae = fit_vae(args, all_loader, "vae_{dataset_type}", dataset_type=dataset_type)
        torch.save(vae, vae_path)
    else:
        print("loading vae...")
        vae = torch.load(vae_path)
    
    def generate_query(query_classifier):
        # pdb.set_trace()
        mc_samp = args.mc_samp

        prior_m = torch.zeros(args.query_size, args.z)
        prior_v = torch.ones(args.query_size, args.z)
        query_z = torch.normal(prior_m, prior_v).requires_grad_(True)

        def simulated_annealing(f, x0, eps=0.05, horizon=500, T=0.2, cooling=0.1):
            x = torch.tensor(x0)
            x_best = torch.tensor(x0)
            # pdb.set_trace()
            f_cur = f(x)
            # f_cur = active_objective(x)
            gamma = cooling ** (1 / horizon)
            for i in trange(horizon):
                x_prop = x + torch.normal(torch.zeros(x.shape), eps * torch.ones(x.shape))
                # pdb.set_trace()
                f_prop = f(x_prop)
                # f_prop = active_objective(x_prop)
                accept_prob = torch.exp((f_cur - f_prop) / T)
                accepted = torch.rand([x.size(0)], device=device) < accept_prob
                x[accepted] = x_prop[accepted]
                f_cur[accepted] = f_prop[accepted]
                T *= gamma
            return x

        def lbfgs(f, x0, horizon=100):
            x0 = x0.clone().detach()
            opt = torch.optim.LBFGS([x0])
            def closure():
                opt.zero_grad()
                loss = -f(x0)
                loss.backward()
                return loss
            for _ in range(horizon):
                opt.step(closure)
            return x0.detach()

        def active_objective(query_z):
            p_z = ut.log_normal(query_z, prior_m, prior_v).exp()
            rec = torch.stack([vae.sample_x_given(query_z) for _ in range(mc_samp)])
            pred_y = query_classifier(rec.reshape(-1, 1, imsize, imsize))
            entropies = td.Categorical(logits=pred_y).entropy().reshape(mc_samp, args.query_size).mean(dim=0)
            return p_z * entropies

        def active_objective_(query_z):
            p_z = ut.log_normal(query_z, prior_m, prior_v).exp()
            rec = torch.stack([vae.sample_x_given(query_z) for _ in range(mc_samp)])
            pred_y = query_classifier(rec.reshape(-1, 1, imsize, imsize))
            entropies = td.Categorical(logits=pred_y).entropy().reshape(mc_samp, args.query_size).mean(dim=0)
            return p_z.mean(), entropies.mean()

        if not args.random:
            query_z = simulated_annealing(active_objective, query_z, horizon=args.sa)
            print(active_objective_(query_z))
        query_x = vae.sample_x_given(query_z).detach()
        return query_x

    active_classifier = initial_classifier
    active_loader = train_loader
    active_dataset = train_subdataset
    for step in tqdm(range(args.horizon)):
        print("step", step)
        query_x = generate_query(active_classifier)
        print("got query_x")

        x_labels = all_classifier(query_x.view(query_x.size(0), 1, imsize, imsize)).argmax(-1)
        new_dataset = torch.utils.data.TensorDataset(query_x.view(query_x.size(0), 1, imsize, imsize), x_labels)
        updated_dataset = torch.utils.data.ConcatDataset([active_dataset, new_dataset])
        active_loader = torch.utils.data.DataLoader(updated_dataset, **train_kwargs)
        active_dataset = updated_dataset
        print("updated dataset")
        classifier = Classifier(args, device, dataset_type)
        print("training classifier...")
        active_classifier = classifier.train(active_loader, test_loader, "new")
        print("testing classifier...")
        results += [classifier.test_model(active_classifier, test_loader)]

        fig, axs = plt.subplots(5, args.query_size // 5)
        for i, ax in enumerate(axs.flat):
            ax.imshow(query_x[i].view(imsize, imsize).detach())
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(f'query-{step}.png')

    print(results)

if __name__ == '__main__':
    main()
