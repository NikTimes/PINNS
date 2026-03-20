import torch
import numpy as np


def semigroup_data(model, ODEsolver, sampler, loss_fn, t_final, num_samples, num_boxes, out_mask, device):
    """
    For each time point in `times`, compute MSE for:
      - direct prediction:  G(y0, t)
      - 2-step iteration:   G(G(y0, t/2), t/2)
      - 3-step iteration:   G(G(G(y0, t/3), t/3), t/3)

    Returns
    -------
    mse_direct, mse_2step, mse_3step : lists of lists (T x N)
    """

    # Initialize Data Arrays
    loss_direct, loss_2step, loss_3step = [], [], []
    
    # Times to evaluate
    times = np.linspace(1, t_final, num_boxes)

    # Collec Initial condition samplers
    y0_samples = sampler(num_samples)

    # Initialize Data Collection
    with torch.no_grad(): 

        # For every final time collect num_samples of MSE data
        for t_val in times:

            d1, d2, d3 = [], [], []

            for y in y0_samples:
                
                # Evaluate true solution
                sol        = ODEsolver((0, t_val), y, t_eval=[t_val]).y[:, -1]
                sol        = sol[out_mask]
                sol_tensor = torch.tensor(sol, dtype=torch.float32).to(device)
                
                # Initial condition input for model
                y0_tensor  = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)

                # Solve using one time step 
                t_tensor = torch.tensor([[t_val]], dtype=torch.float32).to(device)
                p1       = model(y0_tensor, t_tensor)

                # Solve using two timesteps
                t_half   = torch.tensor([[t_val / 2]], dtype=torch.float32).to(device)
                p2       = model(y0_tensor, t_half)
                p2       = model(p2, t_half)

                # Solve using three timesteps
                t_third  = torch.tensor([[t_val / 3]], dtype=torch.float32).to(device)
                p3       = model(y0_tensor, t_third)
                p3       = model(p3, t_third)
                p3       = model(p3, t_third)

                d1.append(loss_fn(p1, sol_tensor).item())
                d2.append(loss_fn(p2, sol_tensor).item())
                d3.append(loss_fn(p3, sol_tensor).item())

            loss_direct.append(d1)
            loss_2step.append(d2)
            loss_3step.append(d3)

    return loss_direct, loss_2step, loss_3step
