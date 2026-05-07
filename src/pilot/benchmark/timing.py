import numpy as np 
import torch
import time


def timing_data(model, ODEsolver, sampler, t_final, num_samples, num_boxes, method, device):

    # Initialize data arrays
    solver_timing = []    
    nn_timing     = []

    # Times to evaluate
    times       = np.linspace(1, t_final, num_boxes)

    # collect initial condition samplers
    y0_samples    = sampler(num_samples)

    # Initialize Data collection
    with torch.no_grad():

        # Collec time to solution for a final time t_eval
        for t_val in times:
            
            solver_time = []
            nn_times    = []

            # for num_samples collect time to solution
            for y in y0_samples:

                # Solve using ODESolver   
                start = time.perf_counter()
                sol   = ODEsolver((0, t_val), y, t_eval=[t_val], method=method)
                end   = time.perf_counter()
                solver_time.append(end - start)

                t_tensor = torch.tensor([[t_val]], dtype=torch.float32).to(device)
                y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)

                # Solve using deeponet
                start     = time.perf_counter()
                model_sol = model(y_tensor, t_tensor)
                end       = time.perf_counter()
                nn_times.append(end - start)
            
            solver_timing.append(solver_time)
            nn_timing.append(nn_times)

    return nn_timing, solver_timing