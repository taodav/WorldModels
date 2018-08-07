import torch

from torch import nn
from models.mdn import MDN

class MemoryRNN(nn.Module):
    def __init__(self, hidden_size, n_gaussians):
        """
        MDN-RNN (Memory RNN). Takes in as input:
        action, previous state, previous hidden state
        :param hidden_size:
        """
        super(MemoryRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_gaussians = n_gaussians

        self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.mdn = MDN(self.hidden_size, self.hidden_size, 10)
