import torch

class Regressor(torch.nn.Module):
    """ Basic regressor network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, in_features: int = 49728+5120, out_features: int = 4) -> None:
        super(Regressor, self).__init__()
        # self.l1 = torch.nn.Linear(in_features, 500)
        # self.l2 = torch.nn.Linear(500, out_features)
        # self.r = torch.nn.ReLU()
        # self.input_layer = torch.nn.Linear(in_features, 500)
        # self.r = torch.nn.ReLU()
        self.linear = torch.nn.Linear(in_features, out_features)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        # # Flatten the input tensor, as x is expected to be of shape [batch_size,in_features]
        # if len(x.shape) > 1:
        #     x = x.flatten()
        
        # return self.linear(self.r(self.input_layer(x)))
        return self.linear(x)