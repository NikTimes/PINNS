try:
    from src.utils import ODEsolver
except ModuleNotFoundError:
    from utils import ODEsolver

import numpy as np 

import torch
from torch.utils.data import IterableDataset, DataLoader


class ODEIterableDataset(IterableDataset):
    """
    Iterable Dataset for on the fly ODE data

    Attributes:
        size         (integer) : Desired size of the Dataset
        system_class  (object) : ODE object with derivative method
        sampler       (object) : Sampler Object with __call__ method
        t_span        (tuple)  : Tuple including initial and end integration times
        method        (String) : solve_ivp numerical integration method
        full_solution (String) : Whether or not to output full solution or solution at time tf

    """

    def __init__(self, 
                 size,
                 system_class,
                 sampler, 
                 t_span,
                 method        = "RK45",
                 full_solution = False):

        self.start      = 0
        self.end        = size
        self.size       = size

        self.system        = system_class
        self.sampler       = sampler
        self.t_span        = t_span
        self.method        = method
        self.full_solution = full_solution

        self.solver     = ODEsolver(system_class)
    
    def __iter__(self):

        # Handle Paralellism 
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            iter_start = self.start
            iter_end   = self.end

        else:
            per_worker = int(np.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id  = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end   = min(iter_start + per_worker, self.end)

        for _ in range(iter_start, iter_end):
            
            # Sample Initial Conditions
            y0     =    self.sampler(1)[0]

            # Sample Random integration end
            t0, tf = self.t_span
            t      = np.random.uniform(t0, tf)

            # Evaluate everywhere
            if self.full_solution:
                sol    = self.solver.solve((t0, tf),   
                                            y0, 
                                            self.method) 
                
                # Store full solution
                y_t    = sol.y[:]
                t_eval = sol.t[:]
                
            
            # Solve only until time t
            else:
                sol    = self.solver.solve((t0, t),     
                                            y0, 
                                            self.method,
                                            t_eval=[t]) # Only Returns solution at t
                
                # Store solution at time t
                y_t    = sol.y[:, -1]
                t_eval = sol.t[:] 
                                        

            I = torch.tensor(y0,    dtype=torch.float32) # Initial Condition
            t = torch.tensor(t_eval,   dtype=torch.float32) # output time 
            y = torch.tensor(y_t,   dtype=torch.float32) # y_I(t)


            yield I, t, y
    
    def __len__(self):
        return self.size


