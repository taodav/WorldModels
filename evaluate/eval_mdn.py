import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import device
from models.mdn import MDN
from datasets.invert_sin import InvertedSineDataset
from utils.helpers import gumbel_sample

def evaluate(dataset, model):
    with torch.no_grad():
        x_test_data = np.linspace(-15, 15, 500)
        # batch size of 500??? this definitely needs to be refactored...
        x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(500, 1)).to(device)

        pi, sigma, mu = model(x_test_tensor)
        pi, sigma, mu = pi.cpu().numpy(), sigma.cpu().numpy(), mu.cpu().numpy()

        # move from torch back to numpy
        k = gumbel_sample(pi)
        indices = (np.arange(500), k)
        rn = np.random.randn(500)
        sampled = rn * sigma[indices] + mu[indices]

        # plot the original data and the test data
        plt.figure(figsize=(8, 8))
        plt.scatter(dataset.x_data, dataset.y_data, alpha=0.2)
        plt.scatter(x_test_data, sampled, alpha=0.2)
        plt.show()


if __name__ == '__main__':
    dataset = InvertedSineDataset(2000)
    model = MDN(1, 20, 5).to(device)
    model.load_state_dict(torch.load('../checkpoints/mdn_model_checkpoint.pt'))

    evaluate(dataset, model)


