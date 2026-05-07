import torch
import torch.nn as nn
from .mlp import General_MLP

class DeepONet(nn.Module):
    """
    Deep Neural Operator designes as in https://arxiv.org/pdf/1910.03193

    Attributes:
        nn (_type_): _description_
    """

    def __init__(self, branch_net, trunk_net, output_size):

        super().__init__()

        self.branch_net = branch_net
        self.trunk_net  = trunk_net

        self.bias       = nn.Parameter(torch.ones(output_size), requires_grad=True)
        self.n_output   = output_size

    def forward(self, u, y):

        # Shape (B, p)
        b = self.branch_net(u)
        t = self.trunk_net(y, final_activation=True)

        # Check that b, and t latent spaces can be divided in chunks 
        p = b.shape[1]
        assert p % self.n_output == 0, f"Latent Dimension p={p} must be divisible by the number of outputs={self.n_output}"
        
        # Determine size that latent space can be 
        chunk_size = p // self.n_output

        # Reshape to (B, n_output, chunk_size) 
        b   = b.view(-1, self.n_output, chunk_size)   # (B, n_output, chunk_size)
        t   = t.view(-1, self.n_output, chunk_size)   # (B, n_output, chunk_size)

        # Sum along (chunk size) to yield shape (B, n_output)
        out = (b*t).sum(dim=-1) + self.bias   

        return out

# ---------------------------------------------------
# Build DeepONet Helper Function
# ---------------------------------------------------


# Build DeepOnet helper method 
def build_deeponet(cfg) -> nn.Module:

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
    
    return DeepONet(branch_net  = branch_net, 
                    trunk_net   = trunk_net, 
                    output_size = cfg["output_size"])
