import torch
import torch.nn as nn

class General_MLP(nn.Module):

    def __init__(self, input_size, output_size, depth, hidden_size, act):

        super().__init__()

        self.layers = nn.ModuleList()
        self.depth  = depth
        self.act    = act

        self.layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(self.depth - 2):

            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x, final_activation):

        for i in range(self.depth - 1):

            x = self.layers[i](x)
            x = self.act(x)

        x = self.layers[-1](x)

        if final_activation:
            return self.act(x)

        else:
            return x

class DeepONet(nn.Module):

    def __init__(self, branch_net, trunk_net):

        super().__init__()

        self.branch_net = branch_net
        self.trunk_net  = trunk_net

        self.bias       = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, u, y):

        b = self.branch_net(u)
        t = self.trunk_net(y, final_activation=True)

        return torch.sum(b * t, dim=1, keepdim=True) + bias