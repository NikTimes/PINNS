import torch
import numpy as np 

def trajectory(model, loader, device):
    
    I, t, y = next(iter(loader))
    I = I[0]
    t = t[0]
    y = y[0]

    with torch.no_grad():
        t_tensor  = t.unsqueeze(1).to(device)
        u0_tensor = I.unsqueeze(0).expand(len(t), -1).to(device)
        model_sol = model(u0_tensor, t_tensor).cpu().numpy()

    t_np = t.cpu().numpy()
    y_np = y.cpu().numpy()

    return t_np, y_np, model_sol

