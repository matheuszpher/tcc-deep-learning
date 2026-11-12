import os
import sys

import argparse
import warnings
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from aeon.datasets import load_classification


def block(in_c, out_c):
    return [nn.Linear(in_c, out_c), nn.ReLU(True)]

class Encoder(nn.Module):
    def __init__(self, input_dim, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential( #
            *block(input_dim, inter_dims[0]),
            *block(inter_dims[0], inter_dims[1]),
            *block(inter_dims[1], inter_dims[2]),
        )
        self.mu_l = nn.Linear(inter_dims[-1], hid_dim)
        self.log_sigma2_l = nn.Linear(inter_dims[-1], hid_dim)

    def forward(self, x):
        e = self.encoder(x)
        mu = self.mu_l(e)
        log_sigma2 = self.log_sigma2_l(e)
        return mu, log_sigma2


class Decoder(nn.Module):
    def __init__(self, input_dim, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            *block(hid_dim, inter_dims[-2]),
            *block(inter_dims[-2], inter_dims[-1]),
            *block(inter_dims[-1], inter_dims[0]),
            nn.Linear(inter_dims[0], input_dim) # Removido Sigmoid para séries contínuas
        )

    def forward(self, z):
        return self.decoder(z)


class VaDE(nn.Module):
    def __init__(self, input_dim, n_clusters, hid_dim=10, device='cuda'):
        super(VaDE, self).__init__()
        self.device = device
        self.n_clusters = n_clusters
        self.hid_dim = hid_dim

        self.encoder = Encoder(input_dim=input_dim, hid_dim=hid_dim)
        self.decoder = Decoder(input_dim=input_dim, hid_dim=hid_dim)

        # Parâmetros da GMM
        self.pi_ = nn.Parameter(torch.FloatTensor(n_clusters).fill_(1.0 / n_clusters), requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(n_clusters, hid_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(n_clusters, hid_dim).fill_(0), requires_grad=True)

    def pre_train(self, dataloader, pre_epoch=30):
        """ Pré-treino do Autoencoder básico + Inicialização do GMM via K-Means/GMM """
        print('Pré-treinando Autoencoder...')
        optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-3)
        loss_fn = nn.MSELoss()
        pretrain_losses = []

        self.train()
        for epoch in range(pre_epoch):
            running_loss = 0.0
            for x, _ in dataloader:
                x = x.to(self.device)
                z, _ = self.encoder(x)
                x_rec = self.decoder(z)
                loss = loss_fn(x_rec, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * x.size(0)
            epoch_loss = running_loss / len(dataloader.dataset)
            pretrain_losses.append(epoch_loss)

        # Extrai representações latentes para inicializar o GMM
        self.eval()
        Z = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                z_mu, _ = self.encoder(x)
                Z.append(z_mu.cpu())

        Z = torch.cat(Z, 0).numpy()
        # Ajusta um GMM no espaço latente pré-treinado
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', random_state=42)
        gmm.fit(Z)

        # Transfere parâmetros do Scikit-Learn para o PyTorch
        self.pi_.data = torch.from_numpy(gmm.weights_).float().to(self.device)
        self.mu_c.data = torch.from_numpy(gmm.means_).float().to(self.device)
        self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).float()).to(self.device)

        return pretrain_losses

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            z_mu, z_sigma2_log = self.encoder(x)
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
            
            # Log p(z|c) + log p(c)
            gamma = torch.log(self.pi_.unsqueeze(0) + 1e-10) + self.gaussian_pdfs_log(z, self.mu_c, self.log_sigma2_c)
            yita = F.softmax(gamma, dim=1)
            return torch.argmax(yita, dim=1).cpu().numpy()

    def ELBO_Loss(self, x, L=1):
        det = 1e-10
        z_mu, z_sigma2_log = self.encoder(x)
        
        # 1. Perda de Reconstrução (MSE para dados contínuos de séries temporais)
        L_rec = 0
        for _ in range(L):
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
            x_rec = self.decoder(z)
            L_rec += F.mse_loss(x_rec, x, reduction='sum') / x.size(0)
        L_rec /= L

        # 2. Distribuições no Espaço Latente (GMM)
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        # p(c|z) responsabilidades via Softmax Estável Numérica
        log_p_z_c = self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)
        log_gamma = torch.log(pi.unsqueeze(0) + det) + log_p_z_c
        gamma = F.softmax(log_gamma, dim=1) # (batch_size, n_clusters)

        # Termos da Divergência KL
        term1 = 0.5 * torch.mean(torch.sum(gamma * torch.sum(
            log_sigma2_c.unsqueeze(0) +
            torch.exp(z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
            (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) / torch.exp(log_sigma2_c.unsqueeze(0)), dim=2), dim=1))

        term2 = torch.mean(torch.sum(gamma * torch.log(pi.unsqueeze(0) / (gamma + det)), dim=1))
        term3 = 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, dim=1))

        Loss = L_rec + term1 - term2 - term3
        return Loss

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.n_clusters):
            G.append(self.gaussian_pdf_log(x, mus[c:c+1, :], log_sigma2s[c:c+1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), dim=1))

    def compute_information_criteria(self, dataloader, n_samples):
        """
        Calcula BIC e AIC aproximados a partir da log-verossimilhança do modelo (Negative ELBO).
        """
        self.eval()
        total_nll = 0.0
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                # Negativo da ELBO aproxima a Log-Likelihood Negativa
                loss = self.ELBO_Loss(x)
                total_nll += loss.item() * x.size(0)

        # Número de parâmetros treináveis no modelo
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        bic = 2 * total_nll + num_params * np.log(n_samples)
        aic = 2 * total_nll + 2 * num_params
        return bic, aic
