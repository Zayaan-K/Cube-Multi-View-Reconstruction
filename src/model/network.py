import torch
from torch import nn


class ReconstructionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 32),    #coordinates
            nn.ReLU(),           
            nn.Linear(32, 32),   
            nn.ReLU(),
            nn.Linear(32, 3),    #vertex
        )

    def forward(self, coordinates):
        return self.network(coordinates)