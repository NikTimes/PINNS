import torch
import torch.nn as nn

class General_MLP(nn.Module):
    """
    Fully connected feedforward neural network (MLP). Inherits from torch nn.Module.

    This class aims to implement a configurable MLP that consists on an input layer
    followed by a sequence of hidden layers with a non-linear activation function and 
    an output layer. The network maps an input from R^{input_size} 
    to an output R^{output_size}

    The aim of this class is to allow the user to costumize the depth, width and 
    activation functions of the MLP with ease. 

    Attributes:
        layers (torch list)   : Contains neural network layers, can be indexed like a python list
        depth  (integer)      : Number of hidden layers in the MLP
        act    (torch method) : Defines the non-linear activation functions
    """

    def __init__(self, input_size, output_size, depth, hidden_size, act):
        """
        Initialization of the MLP. This method builds and stores all relevant
        parameters of the neural network. 

        Args:
            input_size       (integer): Width of the input layer
            output_size      (integer): Width of the output layer
            depth            (integer): Number of hidden layers in the MLP
            hidden_size      (integer): Width of the hidden layers
            act         (torch method): Name of non-linear activation functions
        """
        super().__init__()

        # Storing relevant parameters 
        self.layers = nn.ModuleList()
        self.depth  = depth
        self.act    = act

        # Appending input layer to architecture
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Building hidden layers
        for _ in range(self.depth - 2):

            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Adding output layer 
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x, final_activation=False):
        """
        This method sequentially feeds the data through tall layers of the MLP.
        It is also necessary for training the neural network

        Args:
            x (torch tensor)          : tensor of the same shape as input layer
            final_activation (boolean): determines whether or not output layer has an activation function  

        Returns:
            tensor : MLP output 
        """
        
        for i in range(self.depth - 1):

            x = self.layers[i](x)
            x = self.act(x)

        x = self.layers[-1](x)

        if final_activation:
            return self.act(x)

        else:
            return x

class DeepONet(nn.Module):
    """
    TBC

    Attributes:
        nn (_type_): _description_
    """

    def __init__(self, branch_net, trunk_net):

        super().__init__()

        self.branch_net = branch_net
        self.trunk_net  = trunk_net

        self.bias       = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, u, y):

        b = self.branch_net(u)
        t = self.trunk_net(y, final_activation=True)

        return torch.sum(b * t, dim=1, keepdim=True) + self.bias