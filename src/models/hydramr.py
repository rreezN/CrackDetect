from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn

from src.models.multirocket.multirocket_hydra import MultiRocketHydra
from src.models.hydra.hydra_multivariate import HydraMultivariate



class HydraMR(nn.Module):
    def __init__(
        self,
        num_features: int = 50000,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()

        self.multi_rocket_transformer = MultiRocketHydra(num_features, device)
        self.hydra_transformer = HydraMultivariate(num_features, num_channels=2)
        
    
    def forward(self, X):
        X = X.type(torch.FloatTensor)
        multi_rocket_features = self.multi_rocket_transformer(X.numpy())
        hydra_features = self.hydra_transformer(X)
        multi_rocket_features = torch.tensor(multi_rocket_features)
        
        out_features = torch.hstack((multi_rocket_features, hydra_features))
        
        return out_features