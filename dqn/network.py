import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import msgpack
import os
from .utils import msgpack_numpy_patch
msgpack_numpy_patch()


class Actor_Network(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super().__init__()

        self.device = device

        self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            self.activation,
            nn.Linear(32, 64),
            self.activation,
            nn.Linear(64, 64),
            self.activation,
            nn.Linear(64, 32),
            self.activation,
            nn.Linear(32, output_dim),
        )

        self.to(self.device)

    def forward(self, input):
        return self.net(input)

class Critic_Network(nn.Module):
    def __init__(self, device,  input_dim, output_dim):
        super().__init__()

        self.device = device

        self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
        nn.Linear(input_dim, 32),
        self.activation,
        nn.Linear(32, 64),
        self.activation,
        nn.Linear(64, 64),
        self.activation,
        nn.Linear(64, 32),
        self.activation,
        nn.Linear(32, output_dim),  
    ) 

        self.to(self.device)

    def forward(self, input):
        return self.net(input)