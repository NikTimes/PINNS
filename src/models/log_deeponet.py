import torch
import torch.nn as nn
import numpy as np

from .mlp import General_MLP

class LogDeepONet(nn.Module):
    """
    Deep Neural Operator designes as in https://arxiv.org/pdf/1910.03193
    DeepONet with log1p time transformation for stiff ODEs with large t_span
    https://arxiv.org/pdf/2103.15341

    Attributes:
        nn (_type_): _description_
    """

    def __init__(self, branch_net, trunk_net, output_size, t_span):
        
        super().__init__()

        self.branch_net = branch_net
        self.trunk_net  = trunk_net
        self.bias       = nn.Parameter(torch.zeros(output_size), requires_grad=True)
        self.n_output   = output_size

        # Precompute normalization constants from t_span
        t0, tf          = t_span
        self.log_t_min  = np.log10(t0)
        self.log_t_max  = np.log10(tf)

    def forward(self, u, y):

        # Normalize log10(t) to [0, 1] automatically for any t_span
        y = torch.clamp(y, min=1e-30)
        y = torch.log10(y)
        y = (y - self.log_t_min) / (self.log_t_max - self.log_t_min)

        b   = self.branch_net(u)
        t   = self.trunk_net(y, final_activation=True)

        p          = b.shape[1]
        chunk_size = p // self.n_output

        b   = b.view(-1, self.n_output, chunk_size)
        t   = t.view(-1, self.n_output, chunk_size)
        out = (b * t).sum(dim=-1) + self.bias

        return out
    

# ---------------------------------------------------
# Build DeepONet Helper Function
# ---------------------------------------------------


# Build DeepOnet helper method 
def build_logdeeponet(cfg) -> nn.Module:

    # Currently Available activation functions
    activation_map = {
        "tanh"    : nn.Tanh(),
        "relu"    : nn.ReLU(),
        "sigmoid" : nn.Sigmoid(),
        "gelu"    : nn.GELU()
    }

    # Retrieve Activation function
    activation = activation_map[cfg["activation"]]


    # Initialize Branch and Trunk Networks
    branch_net   = General_MLP(input_size  = cfg["input_size_b"], 
                               output_size = cfg["latent_size"],
                               depth       = cfg["depth"], 
                               hidden_size = cfg["hidden_size"], 
                               act         = activation)

    trunk_net    = General_MLP(input_size  = cfg["input_size_t"], 
                               output_size = cfg["latent_size"],
                               depth       = cfg["depth"], 
                               hidden_size = cfg["hidden_size"], 
                               act         = activation)
    
    return LogDeepONet(branch_net  = branch_net, 
                       trunk_net   = trunk_net, 
                       output_size = cfg["output_size"],
                       t_span      = cfg["t_span"])
