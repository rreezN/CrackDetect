import torch
import torch.nn as nn



class HydraMRRegressor(torch.nn.Module):
    """ Basic regressor network class. 
    
    Parameters:
    ----------
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, in_features: int = 49728+5120, out_features: int = 4) -> None:
        super(HydraMRRegressor, self).__init__()
        
        self.name = 'HydraMRRegressor'
        
        self.input_layer = torch.nn.Linear(in_features, 500)
        self.r = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.linear = nn.Linear(500, out_features)
        
    
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
        x = self.linear(x)
       
        # TODO: Find a proper way to do this 
        # Scale output for now...
        # Targets lie in range [-5, 10] (after standardization)
        x = self.tanh(x)
        x = x * 10
        
        return x
        