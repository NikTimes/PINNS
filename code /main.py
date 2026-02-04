from utils import *

import numpy as np 
import time

# -------------------------------------------------------------------------------
# Hyper Parameters
# -------------------------------------------------------------------------------

# Time Analysis 
num_initial_con = 1000
num_ts          = 10

# Harmonic Oscillator
string_constant = 2
dampening_coeff = 0.1
args1           = [string_constant, dampening_coeff]

# Robertson Model 
k1    = 4e-2
k2    = 3e7
k3    = 1e4
args2 = [k1, k2, k3]


# -------------------------------------------------------------------------------
# Harmonic Oscillator 
# -------------------------------------------------------------------------------

def oscillator_time_boxplot(args, num_initial_con, num_ts):

    oscillator = harm_osc(args1) 
    y0         = 2*initial_sampler(num_initial_con, 2) - 1
    solver     = ODEsolver(oscillator).solve

    ts          = np.linspace(1.0, 100, num_ts)
    timing_data = []

    for t in ts:
        
        times  = []

        for y in y0:

            start = time.perf_counter()
            sol   = solver((0, t), y)
            end   = time.perf_counter()

            times.append(end-start)
        
        timing_data.append(times)
        


    plt.boxplot(timing_data, positions=ts, widths=8, showfliers=False)
    plt.xlabel("End time t")
    plt.ylabel("Time to solution (s)")
    plt.title("Time-to-solution vs end time for harmonic oscillator")
    plt.show()

# -------------------------------------------------------------------------------
# Roberston Model 
# -------------------------------------------------------------------------------

def robertson_time_boxplot(args, num_initial_con, num_ts):

    robertson = Robertson(args2)
    y0        = np.random.dirichlet(alpha=[1, 1, 1], size=num_initial_con)
    solver    = ODEsolver(robertson).solve

    ts          = np.linspace(1000.0, 1e6, num_ts)
    timing_data = []

    for t in ts:
        
        times  = []

        for y in y0:

            start = time.perf_counter()
            sol   = solver((0, t), y, method="BDF")
            end   = time.perf_counter()

            times.append(end-start)
        
        timing_data.append(times)

    plt.boxplot(timing_data, positions=ts, widths=1e5, showfliers=False)
    plt.xlabel("End time t")
    plt.ylabel("Time to solution (s)")
    plt.title("Time-to-solution vs end time for Roberston Model")

    plt.xticks(rotation=20)  

    plt.show()

oscillator_time_boxplot(args1, num_initial_con, num_ts)