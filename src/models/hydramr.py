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
                 batch_norm: bool = False,
                 kpi_means: torch.Tensor = None,
                 kpi_stds: torch.Tensor = None,
                 kpi_mins: torch.Tensor = None,
                 kpi_maxs: torch.Tensor = None,
                 ) -> None:
        super(HydraMRRegressor, self).__init__()
        
        self.name = name
        self.kpi_means = nn.Parameter(kpi_means) if kpi_means is not None else nn.Parameter(torch.zeros(out_features))
        self.kpi_stds = nn.Parameter(kpi_stds) if kpi_stds is not None else nn.Parameter(torch.ones(out_features))
        self.kpi_mins = nn.Parameter(kpi_mins) if kpi_mins is not None else nn.Parameter(torch.zeros(out_features))
        self.kpi_maxs = nn.Parameter(kpi_maxs) if kpi_maxs is not None else nn.Parameter(torch.ones(out_features))
        
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
    
import torch
import torch.nn as nn


class HydraMRRegressor_old(torch.nn.Module):
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
                 name: str = 'HydraMRRegressor_old') -> None:
        super(HydraMRRegressor_old, self).__init__()
        
        self.name = name
        
        self.input_layer = torch.nn.Linear(in_features, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.r = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.linear = nn.Linear(hidden_dim, out_features)
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
        x = self.r(self.hidden(x))  # Pass through hidden layer
        x = self.dropout(x)  # Apply dropout after hidden layer
        x = self.linear(x)
       
        # TODO: Find a proper way to do this 
        # Scale output for now...
        # Targets lie in range [-5, 10] (after standardization)
        # x = self.tanh(x)
        # x = x * 10
        
        return x