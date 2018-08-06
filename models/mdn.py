import torch

import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F


class MDN(nn.Module):
    def __init__(self, n_input, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians

        self.w_b = nn.Linear(n_input, n_hidden)
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, input_tensor):
        z_h = F.tanh(self.w_b(input_tensor))
        pi = F.softmax(self.z_pi(z_h), dim=1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)

        return pi, sigma, mu

