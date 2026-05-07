import torch
import torch.nn as nn
import wandb

class RobertsonPhysicsLoss(nn.Module):
    
    def __init__(self, k1, k2, k3, yscale,
                 lambda_data= 1.0,
                 lambda_ode = 1.0,
                 lambda_cons= 1.0):
                 
        super().__init__()
        
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        
        self.register_buffer("yscale",torch.tensor(yscale, dtype=torch.float32))
        
        self.lambda_data = lambda_data
        self.lambda_ode  = lambda_ode
        self.lambda_cons = lambda_cons
        
        self._t  = None  # set before each forward call
        self._y0 = None

    def set_inputs(self, y0, t):
        
        self._y0 = y0
        self._t  = t

    def forward(self, pred, target):
        
        # 1. Data loss
        data_loss = torch.mean(((pred - target) / self.yscale) ** 2)

        # 2. ODE residual
        t    = self._t.requires_grad_(True)
        dydt = []

        
        for k in range(3):
            
            grad_k = torch.autograd.grad(
                            outputs = pred[:, k].sum(),
                            inputs  = t,
                            create_graph = True,)[0]
            
            dydt.append(grad_k)
        
        dydt = torch.cat(dydt, dim=1)

        y1, y2, y3 = pred[:, 0], pred[:, 1], pred[:, 2]
        r1 = dydt[:, 0] - (-self.k1*y1 + self.k3*y2*y3)
        r2 = dydt[:, 1] - ( self.k1*y1 - self.k3*y2*y3 - self.k2*y2**2)
        r3 = dydt[:, 2] - ( self.k2*y2**2)
        ode_loss = torch.mean(torch.stack([r1, r2, r3], dim=1) ** 2)

        # 3. Conservation loss
        cons_loss = torch.mean((pred.sum(dim=1) - 1.0) ** 2)

        return (self.lambda_data * data_loss
              + self.lambda_ode  * ode_loss
              + self.lambda_cons * cons_loss)