# Copyright (c) 2021 Rui Shu
import numpy as np
import torch
import utils as ut
from models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

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
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        m_mixture, v_mixture = prior

        m, v = self.enc(x)
        z = ut.sample_gaussian(m, v) 
        logits = self.dec(z)
        rec = -ut.log_bernoulli_with_logits(x, logits)
        rec = torch.mean(rec)

        kl = ut.log_normal(z, m, v) - ut.log_normal_mixture(z, m_mixture, v_mixture)
        kl = torch.mean(kl)

        nelbo = kl + rec

        # NELBO: 98.7930908203125. KL: 17.768733978271484. Rec: 81.02436065673828

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
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        m_mixture, v_mixture = prior

        m, v = self.enc(x)
        z = ut.sample_gaussian(m, v) 
        logits = self.dec(z)
        rec = -ut.log_bernoulli_with_logits(x, logits)
        rec = torch.mean(rec)

        kl = ut.log_normal(z, m, v) - ut.log_normal_mixture(z, m_mixture, v_mixture)
        kl = torch.mean(kl)

        # compute IWAE
        x_dup = ut.duplicate(x, iw)
        m, v = self.enc(x_dup)
        zs = ut.sample_gaussian(m, v) 
        logits = self.dec(zs)

        p_theta_x_z = ut.log_bernoulli_with_logits(x_dup, logits)
        q_phi_z_x = ut.log_normal(zs, m, v) 
        p_theta_z = ut.log_normal_mixture(zs, m_mixture, v_mixture)
        
        niwae_sampled = p_theta_x_z - q_phi_z_x + p_theta_z
        niwae = -ut.log_mean_exp(niwae_sampled.reshape(iw, -1), dim=0)
        niwae = torch.mean(niwae)

        # NELBO: 98.88081512451172. KL: 17.737916946411133. Rec: 81.04288482666016
        # Negative IWAE-1: 98.85325622558594
        # Negative IWAE-10: 96.37853240966797
        # Negative IWAE-100: 95.31634521484375
        # Negative IWAE-1000: 94.9302749633789
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
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
