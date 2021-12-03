# Copyright (c) 2021 Rui Shu
import numpy as np
import torch
import torch.nn.functional as F
import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F

class Classifier(nn.Module):
    def __init__(self, y_dim, dataset_type):
        super().__init__()
        self.y_dim = y_dim
        if dataset_type == 'mnist':
            self.net = nn.Sequential(
                nn.Linear(784, 300),
                nn.ELU(),
                nn.Linear(300, 300),
                nn.ELU(),
                nn.Linear(300, y_dim),
            )
        elif dataset_type == 'dogs':
            self.net = nn.Sequential(
                nn.Linear(64, 3072),
                nn.ELU(),
                nn.Linear(3072, 3072),
                nn.ELU(),
                nn.Linear(3072, y_dim),
            )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, z_dim, y_dim=0, dataset_type='mnist'):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        if dataset_type == 'mnist':
            self.net = nn.Sequential(
                nn.Linear(784 + y_dim, 300),
                nn.ELU(),
                nn.Linear(300, 300),
                nn.ELU(),
                nn.Linear(300, 2 * z_dim),
            )
        elif dataset_type == 'dogs':
            self.net = nn.Sequential(
                nn.Linear(64 + y_dim, 3072),
                nn.ELU(),
                nn.Linear(3072, 3072),
                nn.ELU(),
                nn.Linear(3072, 2 * z_dim),
            )

    def forward(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0, dataset_type='mnist'):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        if dataset_type == 'mnist':
            self.net = nn.Sequential(
                nn.Linear(z_dim + y_dim, 300),
                nn.ELU(),
                nn.Linear(300, 300),
                nn.ELU(),
                nn.Linear(300, 784)
            )
        elif dataset_type == 'dogs':
            self.net = nn.Sequential(
                nn.Linear(z_dim + y_dim, 3072),
                nn.ELU(),
                nn.Linear(3072, 3072),
                nn.ELU(),
                nn.Linear(3072, 12544)
            )

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)

# class Classifier(nn.Module):
#     def __init__(self, y_dim):
#         super().__init__()
#         self.y_dim = y_dim
#         self.net = nn.Sequential(
#             nn.Linear(784, 300),
#             nn.ReLU(),
#             nn.Linear(300, 300),
#             nn.ReLU(),
#             nn.Linear(300, y_dim)
#         )

#     def forward(self, x):
#         return self.net(x)
