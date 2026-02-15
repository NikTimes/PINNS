from utils import *
import numpy as np 

import torch
from torch.utils.data import IterableDataset, DataLoader


class ODEIterableDataset(IterableDataset):
    """TBC


    """

    def __init__(
        self, 
        system_class,
        y0_sampler,
        n_samples,
        t_span,
        method="RK45",
        ):
        
        super().__init__()
        self.start = 0
        self.end   = n_samples

        self.system     = system_class
        self.y0_sampler = y0_sampler
        self.t_span     = t_span
        self.method     = method

        self.solver     = ODEsolver(system_class)


    def __iter__(self):
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

            # Samples Initial Condition 
            y0     = self.y0_sampler(1)[0]

            # Samples Random Time
            t0, tf = self.t_span
            t = np.random.uniform(t0, tf)
            
            # Solve only until t
            sol    = self.solver.solve((t0, t), 
                                        y0, 
                                        self.method, 
                                        t_eval=[t]) # Only Returns Solution at t 
            
            y_t = sol.y[:, -1]

            I = torch.tensor(y0, dtype=torch.float32)  # Initial Condition
            t = torch.tensor([t], dtype=torch.float32) # output time 
            y = torch.tensor(y_t, dtype=torch.float32) # y_I(t)

            yield I, t, y





"""
Testing the dataset Class

k = 2.0
c = 0.5
system = harm_osc([k, c])

sampler = LatinHypercubeSampler(
    dimensions=2,
    lows=[-1.0, -1.0],
    highs=[1.0, 1.0]
)


dataset = ODEIterableDataset(
    system_class=system,
    y0_sampler=sampler,
    n_samples=100,
    t_span=(0.0, 10),
    method="RK45"
)


it = iter(dataset)  # create the generator

for i in range(100):
    I, t, y = next(it)  # generate one sample

    print(f"\nSample {i+1}")
    print("Initial condition I:", I)
    print("Time t:", t)
    print("Solution y(t):", y)
"""