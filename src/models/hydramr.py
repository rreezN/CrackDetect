import torch
import torch.nn as nn


class HydraMRRegressor(torch.nn.Module):
    """ Basic regressor network class. 
    
    Parameters:
    ----------
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, in_features: int = 49728+5120, out_features: int = 4, name: str = 'HydraMRRegressor') -> None:
        super(HydraMRRegressor, self).__init__()
        
        self.name = name
        
        self.input_layer = torch.nn.Linear(in_features, 30)
        self.dropout = nn.Dropout(0.5)
        self.hidden = torch.nn.Linear(30, 30)
        self.r = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.linear = nn.Linear(30, out_features)
        # TODO add a hidden layer. 500 might be to big!
        # regularization is needed
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
        x = self.r(self.input_layer(x))
        x = self.dropout(x)  # Apply dropout after activation
        x = self.r(self.hidden_layer(x))  # Pass through hidden layer
        x = self.dropout(x)  # Apply dropout after hidden layer
        x = self.linear(x)
       
        # TODO: Find a proper way to do this 
        # Scale output for now...
        # Targets lie in range [-5, 10] (after standardization)
        x = self.tanh(x)
        x = x * 10
        
        return x