from utils import *
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

    """

    def __init__(self, 
                 size,
                 system_class,
                 sampler, 
                 t_span,
                 method="RK45"):

        self.start   = 0
        self.end     = size
        self.size    = size

        self.system  = system_class
        self.sampler = sampler
        self.t_span  = t_span
        self.method  = method

        self.solver  = ODEsolver(system_class)
    
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

            # Solve only until t
            sol    = self.solver.solve((t0, tf), 
                                        y0, 
                                        self.method,
                                        t_eval=[t]) # Only Returns solution at t
            
            # Extract final position 
            y_t = sol.y[:, -1][0] 

            I = torch.tensor(y0,    dtype=torch.float32)  # Initial Condition
            t = torch.tensor([t],   dtype=torch.float32) # output time 
            y = torch.tensor([y_t], dtype=torch.float32) # y_I(t)


            yield I, t, y
    
    def __len__(self):
        return self.size


"""
k = 2.0
c = 0.5
system = harm_osc([k, c])

sampler = LatinHypercubeSampler(
    dimensions=2,
    lows      = [-1.0, -1.0],
    highs     = [1.0, 1.0]
)

"""