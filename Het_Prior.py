#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:07:39 2022

@author: alain
"""

import torch
from torch import nn
from torch import optim



# regularizer used in proximal-relational autoencoder
class Prior(nn.Module):
    def __init__(self, data_size: list,mean_var=3):
        super(Prior, self).__init__()
        # data_size = [num_component, z_dim]
        self.data_size = data_size
        self.dim = data_size[1]
        self.number_components = data_size[0]
        self.output_size = data_size[1]
        self.mu_init = torch.randn(data_size)*mean_var
        self.mu = self.mu_init.clone()
        self.logvar = torch.ones(data_size)
        self.mu_temp = torch.zeros(data_size)
        self.n_update = 0
        self.mu_local = None
    def forward(self):
        return self.mu, self.logvar
    
    def sampling_gmm(self,num_sample):
        std = torch.exp(0.5 * self.logvar)
        n = int(num_sample / self.mu.size(0)) + 1
        for i in range(n):
            eps = torch.randn_like(std)
            if i == 0:
                samples = self.mu + eps * std
            else:
                samples = torch.cat((samples, self.mu + eps * std), dim=0)
        return samples[:num_sample, :]
    def init_mu_temp(self):
        self.mu_temp = torch.zeros(self.data_size)
        self.n_update = 0



    def sampling_gaussian(self,num_sample,mean, logvar):
        std = torch.exp(0.5 * logvar)
        for i in range(num_sample):
            eps = torch.randn_like(std)
            if i == 0:
                samples = (mean + eps * std).reshape(1,self.dim)
            else:
                aux = (mean + eps * std).reshape(1,self.dim)

                samples = torch.cat((samples, aux), dim=0)
      
        return samples


if __name__ == '__main__':

        prior = Prior(data_size=[2,2])
        z_mu, z_logvar = prior()
        prior.logvar[0]=torch.Tensor([0,0])
        z_samples = prior.sampling_gaussian(10000,prior.mu[0], prior.logvar[0])
        import numpy as np
        m = z_samples.mean(dim=0)
        Cov = (z_samples - m).T @(z_samples - m)/(z_samples.shape[0]**2) 
        import matplotlib.pyplot as plt

        # n, bins, patches = plt.hist(z_samples.numpy(), 50, density=True, facecolor='g', alpha=0.75)

        plt.scatter(z_samples[:,0],z_samples[:,1])