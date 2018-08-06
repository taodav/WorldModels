import torch
import numpy as np

def gaussian_distribution(y, mu, sigma):
    one_div_sqrt_pi = 1.0 / (np.sqrt(2.0 * np.pi))
    result = (y.expand_as(mu) - mu) / sigma
    result = -0.5 * (result.pow(2))
    return (torch.exp(result) / sigma) * one_div_sqrt_pi

def mdn_loss_func(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)
