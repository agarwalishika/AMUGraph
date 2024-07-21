# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : semi_vae.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description:
"""

from typing import List
import torch.nn as nn
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class PIControl:
    """Feedback System, PI Control"""

    def __init__(self):
        self.i_p = 0.0
        self.beta = 0.0
        self.error = 0.0

    def pi(self, err, beta_min=1, n=1, kp=1e-2, ki=1e-4):
        beta_i = None
        for i in range(n):
            p_i = kp / (1.0 + np.exp(err))
            i_i = self.i_p - ki * err

            if self.beta < 1.0:
                i_i = self.i_p
            beta_i = p_i + i_i + beta_min

            self.i_p = i_i
            self.beta = beta_i
            self.error = err

            if beta_i < beta_min:
                beta_i = beta_min
        return beta_i

def _loss_semi_vae(x: torch.Tensor, x_hat: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor):
    """Loss function of semi-supervised vae"""

    n = np.prod(x.shape[1:])

    x, x_hat = x.view(-1, n), x_hat.view(-1, n)

    x_n = y[:, 0].view(-1,1) * x
    x_hat_n = y[:, 0].view(-1,1) * x_hat
    #x_hat_n = torch.mean(x_hat_n, axis=0).repeat(x_n.shape[0], 1)
    mu_n = y[:, 0].view(-1,1) * mu
    log_var_n = y[:, 0].view(-1,1) * log_var

    x_a = y[:, 1].view(-1,1) * x
    x_hat_a = y[:, 1].view(-1,1) * x_hat

    if (torch.max(x_hat_n) <= 1 and torch.min(x_hat_n) >= 0) and (torch.max(x_n) <= 1 and torch.min(x_n) >= 0):
        bce_loss_n = F.binary_cross_entropy(x_hat_n, x_n, reduction='sum') / x_hat_n.shape[0] if x_hat_n.shape[0] else 0
    else:
        bce_loss_n = F.binary_cross_entropy_with_logits(x_hat_n, x_n, reduction='sum') / x_hat_n.shape[0] if x_hat_n.shape[0] else 0
    
    if (torch.max(x_hat_a) <= 1 and torch.min(x_hat_a) >= 0) and (torch.max(x_a) <= 1 and torch.min(x_a) >= 0):
        bce_loss_a = F.binary_cross_entropy(x_hat_a, x_a, reduction='sum') / x_hat_a.shape[0] if x_hat_a.shape[0] else 0
    else:
        bce_loss_a = F.binary_cross_entropy_with_logits(x_hat_a, x_a, reduction='sum') / x_hat_a.shape[0] if x_hat_a.shape[0] else 0

    bce_loss = bce_loss_n - bce_loss_a
    kld_loss = -0.5 * torch.sum(1 + log_var_n - mu_n.pow(2) - log_var_n.exp()) / log_var_n.shape[0] if log_var_n.shape[0] else torch.Tensor([0])
    kld_loss = kld_loss.to('cuda')

    #with open('bce_loss.txt', 'a+') as f:
    #    f.write(f'bce: {bce_loss_n} vs{bce_loss_a} \t kld: {kld_loss}\n')
    return bce_loss, kld_loss

def _sample(mu: torch.Tensor, log_var: torch.Tensor):
    """Sample function of vae"""

    std = torch.exp(0.5 * log_var)
    return torch.randn_like(std).mul(std).add_(mu)


class SemiVAE(nn.Module):
    """Semi-supervised variational auto-encoder"""

    def __init__(self, in_dim: List[int], latent_dim: int, hidden: List[int], **kwargs):
        """

        :param in_dim: input dimension
        :param latent_dim: latent var dimension
        :param hidden: hidden layers of encoder and decoder layers
        :param kwargs:
        """

        super(SemiVAE, self).__init__()

        self.in_dim = in_dim

        n = len(hidden)
        self.encode_layers = nn.ModuleList()
        self.encode_layers.append(nn.Linear(in_features=np.prod(in_dim), out_features=hidden[0]))
        for i in range(n - 1):
            self.encode_layers.append(nn.Linear(in_features=hidden[i], out_features=hidden[i + 1]))

        self.mu = nn.Linear(in_features=hidden[-1], out_features=latent_dim)
        self.log_var = nn.Linear(in_features=hidden[-1], out_features=latent_dim)

        self.latent_layer = nn.Linear(in_features=latent_dim, out_features=hidden[-1])
        self.decoder_layers = nn.ModuleList()
        for i in range(n - 1):
            self.decoder_layers.append(nn.Linear(in_features=hidden[n - i - 1], out_features=hidden[n - i - 2]))
        self.decoder_layers.append(nn.Linear(in_features=hidden[0], out_features=np.prod(in_dim)))

        self.kwargs = kwargs

    def encoder(self, x: torch.Tensor):
        """
        semi-vae encoder

        :param x: support multiple types of input, e.g.,
                    Single KPI time series: [N,]
                    Multiple KPI time series: [M, N], ...
        :return:
        """
        try:
            x = torch.flatten(x, start_dim=1)
        except:
            hi = 9 # do nothing
        for fc in self.encode_layers:
            x = F.relu(fc(x))

        x_mu = self.mu(x)
        x_log_var = nn.Softplus()(self.log_var(x))
        return x_mu, x_log_var

    def decoder(self, x: torch.Tensor):
        """
        semi-vae decoder

        :param x: latent var shape=[N, latent_dim]
        :return:
        """

        x = self.latent_layer(x)
        for fc in self.decoder_layers:
            x = fc(F.relu(x))

        # Since the input is in a range of [0, 1], sigmoid is adopted.
        x = torch.sigmoid(x)

        # [N, np.prod(in_dim)] -> [N, D1, D2, ...]
        x = x.view([x.shape[0]] + self.in_dim)
        return x

    def forward(self, x: torch.Tensor):
        """forward propagation"""

        x_mu, x_log_var = self.encoder(x)
        z = _sample(x_mu, x_log_var)
        x_hat = self.decoder(z)
        return x_hat, x_mu, x_log_var

    def fit(self, x: np.ndarray, y: np.ndarray, epochs=50, batch_size=256, lr=1e-3, desired_kl=10, kp=1e-2, ki=1e-4):
        """semi-vae model training"""

        data_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=lr)

        pi = PIControl()

        for _ in tqdm(range(epochs), position=2, desc='semi_vae'):
            for x_batch, y_batch in data_loader:
                opt.zero_grad()
                x_hat, mu, log_var = self.forward(x_batch)
                bce_loss, kld_loss = _loss_semi_vae(x_batch, x_hat, y_batch, mu, log_var)
                try:
                    beta = pi.pi(desired_kl - kld_loss.item(), kp=kp, ki=ki)
                except:
                    hi = 9
                    _loss_semi_vae(x_batch, x_hat, y_batch, mu, log_var)
                loss = bce_loss + beta * kld_loss
                loss.backward(retain_graph=True)
                opt.step()
        return

    def predict(self, x: np.ndarray):
        """semi-vae model predicting"""

        x = torch.Tensor(x)
        x_hat, x_mu, x_log_var = self.forward(x)
        return x_hat, x_mu, x_log_var