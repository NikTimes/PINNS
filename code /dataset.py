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
        t_eval=None 
        ):
        
        super().__init__()
        self.start = 0
        self.end   = n_samples

        self.system     = system_class
        self.y0_sampler = y0_sampler
        self.t_eval     = t_eval
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

            t_span = self.t_span
            t_eval = self.t_eval
            y0     = self.y0_sampler(1)[0]
            method = self.method

            sol    = self.solver.solve(t_span, y0, method, t_eval=t_eval)

            t      = torch.tensor(sol.t, dtype=torch.float32)
            y      = torch.tensor(sol.y.T, dtype=torch.float32)

        yield t, y



"""
This is just a test that the dataset class is working as intended

osc_sampler = LatinHypercubeSampler(2, [-1, -1], [1, 1])
rob_sampler   = DirichletSampler(alpha=[1, 1, 1])

k1          = 4e-2
k2          = 3e7
k3          = 1e4
args2       = [k1, k2, k3]

dataset = ODEIterableDataset(
    Robertson(args2), 
    rob_sampler,
    10,
    (0, 100),
    method = "BDF"
)


for _ in range(10):
    it = iter(dataset)
    t, y = next(it)
    plt.xscale("log")
    plt.plot(t, y)

plt.show()
"""