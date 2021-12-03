# Copyright (c) 2021 Rui Shu

import torch
import utils as ut
from models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2, dataset_type='mnist'):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, dataset_type=dataset_type)
        self.dec = nn.Decoder(self.z_dim, dataset_type=dataset_type)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        m, v = self.enc(x)
        z = ut.sample_gaussian(m, v) 
        logits = self.dec(z)
        rec = -ut.log_bernoulli_with_logits(x, logits)
        rec = torch.mean(rec)

        kl = ut.kl_normal(m, v, self.z_prior_m , self.z_prior_v)
        kl = torch.mean(kl)

        nelbo = kl + rec

        # NELBO: 100.48635864257812. KL: 19.431188583374023. Rec: 81.05512237548828

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################

        m, v = self.enc(x)
        z = ut.sample_gaussian(m, v) 
        logits = self.dec(z)
        rec = -ut.log_bernoulli_with_logits(x, logits)
        rec = torch.mean(rec)

        kl = ut.kl_normal(m, v, self.z_prior_m , self.z_prior_v)
        kl = torch.mean(kl)

        # compute IWAE
        x_dup = ut.duplicate(x, iw)
        m, v = self.enc(x_dup)
        zs = ut.sample_gaussian(m, v) 
        logits = self.dec(zs)

        p_theta_x_z = ut.log_bernoulli_with_logits(x_dup, logits)
        q_phi_z_x = ut.log_normal(zs, m, v) 
        p_theta_z = ut.log_normal(zs, self.z_prior_m , self.z_prior_v)
        
        niwae_sampled = p_theta_x_z - q_phi_z_x + p_theta_z
        niwae = -ut.log_mean_exp(niwae_sampled.reshape(iw, -1), dim=0)
        niwae = torch.mean(niwae)

        # NELBO: 100.56014251708984. KL: 19.431188583374023. Rec: 81.12892150878906
        # Negative IWAE-1: 100.47402954101562
        # Negative IWAE-10: 97.9261703491211
        # Negative IWAE-100: 97.09532928466797
        # Negative IWAE-1000: 96.47144317626953

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
