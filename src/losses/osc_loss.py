import torch
import torch.nn as nn 

class osc_loss(nn.Module):

    def __init__(self, system):
        super().__init__()
        # Store system object as attribute

        # system is a plain Python object, assign directly
        self.system = system
        
        # Register k and c as buffers so they move with .to(device) automatically
        self.register_buffer('k', torch.tensor(system.k, dtype=torch.float32))
        self.register_buffer('c', torch.tensor(system.c, dtype=torch.float32))


    def forward(self, pred):
        x, v = pred
        dxdt, dvdt = self.system.derivative(None, [x, v])
        
        return dvdt + self.c * dxdt + self.k * x




