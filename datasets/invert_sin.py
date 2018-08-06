import torch
import numpy as np
from torch.utils.data import Dataset
from utils import device


class InvertedSineDataset(Dataset):
    def __init__(self, n_samples):
        super(InvertedSineDataset, self).__init__()

        self.n_samples = n_samples
        self.data = self.generate_data()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def generate_data(self):
        epsilon = np.random.normal(size=self.n_samples)
        x_data = np.random.uniform(-10.5, 10.5, self.n_samples)
        y_data = 7 * np.sin(0.75 * x_data) + 0.5 * x_data + epsilon

        x_data = torch.from_numpy(x_data.reshape(self.n_samples, 1)).float().to(device)
        y_data = torch.from_numpy(y_data.reshape(self.n_samples, 1)).float().to(device)
        return list(zip(x_data, y_data))