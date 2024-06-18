import torch
import torch.nn as nn


class HydraMRRegressor(torch.nn.Module):
    """ Basic regressor network class. 
    
    Parameters:
    ----------
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, 
                 in_features: int = 49728+5120, 
                 out_features: int = 4, 
                 hidden_dim: int = 100,
                 dropout: float = 0.5,
                 name: str = 'HydraMRRegressor',
                 model_depth: int = 1,
                 batch_norm: bool = False
                 ) -> None:
        super(HydraMRRegressor, self).__init__()
        
        self.name = name
        self.tanh = torch.nn.Tanh()
        
        # Input layer 
        self.input_layer = torch.nn.Linear(in_features, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.r = torch.nn.ReLU()
        self.linear = nn.Linear(hidden_dim, out_features)
        layers = [self.input_layer]
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
        layers.append(self.r)
        layers.append(self.dropout)
        
        # Hidden layers
        for _ in range(model_depth):
            if batch_norm:
                layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), self.r, self.dropout])
            else:
                layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), self.r, self.dropout])
        
        # Output layer
        layers.append(self.linear)
        
        # Create the network
        self.net = nn.Sequential(*layers)

        # Permutation test on the 30 raw signals. 
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Parameters:
        ----------
            x: input tensor expected to be of shape [N,in_features]

        Returns:
        ----------
            Output tensor with shape [N,out_features]

        """
        x = self.net(x)
       
        # TODO: Find a proper way to do this 
        # Scale output for now...
        # Targets lie in range [-5, 10] (after standardization)
        # x = self.tanh(x)
        # x = x * 10
        
        return x