import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_length, output_length=1):
        super(Regressor, self).__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.model = nn.Sequential(
            nn.Linear(self.input_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_length)
        )
        
    def forward(self, x):
        return self.model(x)