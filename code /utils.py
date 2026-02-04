import numpy as np 
import matplotlib.pyplot as plt 

from scipy.integrate import solve_ivp
from scipy.stats import qmc


# -------------------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------------------

class harm_osc():

    def __init__(self, args):

        self.k = args[0]
        self.c = args[1]


    def derivative(self, t, y):

        x, v = y
        dxdt = v
        dvdt = - self.k*x - self.c*v

        return np.array([dxdt, dvdt])
    
class Robertson():
    def __init__(self, args):
        
        self.k1 = args[0]
        self.k2 = args[1]
        self.k3 = args[2]

    def derivative(self, t, y):

        y1, y2, y3 = y

        dy1 = -self.k1*y1 + self.k2*y2*y3
        dy2 =  self.k1*y1 - self.k2*y2*y3 - self.k3*y2**2
        dy3 =  self.k3*y2**2

        return np.array([dy1, dy2, dy3])


class ODEsolver(): 
    
    def __init__(self, system):
        self.system      = system
        self.derivatives = system.derivative
    
    def solve(self, t_span, y0, method="RK45", t_eval= None, dense_output=False, vectorized=False):

        sol = solve_ivp(
                self.derivatives, 
                t_span=t_span,
                y0 = y0,
                method=method,
                t_eval=t_eval,
                dense_output=dense_output, 
                vectorized=vectorized,
                rtol=1e-6,
                atol=1e-9,
            )

        return sol

# -------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------

def initial_sampler(num_samples, dimension):
    
    sampler = qmc.LatinHypercube(d=dimension)
    samples = sampler.random(num_samples)

    return samples