from model import *
from utils import *
from dataset import ODEIterableDataset

import torch
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm 


# -------------------------------------------------------------------------------
# 0. Hyper Parameters 
# -------------------------------------------------------------------------------

# Training Parameters 
num_samples_per_epoch = 1_000
num_epochs            = 32

batch_size            = 32
num_workers           = 4
learning_rate         = 0.016

# Loss Functions 

loss_fn     = torch.nn.MSELoss()

# Harmonic Oscillator 

string_constant = 2
dampening_coeff = 0.1
args1           = [string_constant, dampening_coeff]

osc_sampler     = LatinHypercubeSampler(dimensions = 2, 
                                        lows       = [-1, -1], 
                                        highs      = [1, 1])

osc_object      = harm_osc(args1)
osc_n_samples   = num_samples_per_epoch
osc_t_span      = (0, 100)

osc_dataset     = ODEIterableDataset(system_class = osc_object, 
                                     y0_sampler   = osc_sampler, 
                                     n_samples    = osc_n_samples, 
                                     t_span       = osc_t_span,
                                     method       = "RK45")

# Robertson Model 

k1            = 4e-2
k2            = 3e7
k3            = 1e4
args2         = [k1, k2, k3]

rob_sampler   = DirichletSampler(alpha = [1, 1, 1])

rob_object    = Robertson(args2) 
rob_n_samples = num_samples_per_epoch
rob_t_span    = (0, 1e6)

rob_dataset   = ODEIterableDataset(system_class = rob_object, 
                                   y0_sampler   = rob_sampler, 
                                   n_samples    = rob_n_samples, 
                                   t_span       = rob_t_span,
                                   method       = "BDF")


# -------------------------------------------------------------------------------
# 1. Retrieving data 
# -------------------------------------------------------------------------------

# Harmonic Oscillator 

osc_loader = DataLoader(dataset     = osc_dataset, 
                        batch_size  = batch_size,
                        num_workers = num_workers)

# Roberston Model

rob_loader = DataLoader(dataset     = rob_dataset, 
                        batch_size  = batch_size,
                        num_workers = num_workers)


# -------------------------------------------------------------------------------
# 2. Models
# -------------------------------------------------------------------------------

# Harmonic Oscillator 

# Define branch and trunk network parameters 
osc_input_size  = 3
osc_output_size = 1
osc_depth       = 2
osc_hidden_size = 32

# Initialize Branch and Trunk Nets 
osc_branch_net = General_MLP(input_size  = osc_input_size, 
                             output_size = osc_output_size,
                             depth       = osc_depth, 
                             hidden_size = osc_hidden_size, 
                             act         = nn.Tanh)

osc_trunk_net = General_MLP(input_size   = osc_input_size, 
                             output_size = osc_output_size,
                             depth       = osc_depth, 
                             hidden_size = osc_hidden_size, 
                             act         =nn.Tanh)

# Initialize DeepONet 
osc_deepONet = DeepONet(branch_net = osc_branch_net, 
                        trunk_net  = osc_trunk_net)

# -------------------------------------------------------------------------------
# 3. WandB config
# -------------------------------------------------------------------------------

wandb.init(
    project="First DeepONet trial",
    config={
        "epochs"        : num_epochs,
        "batch size"    : batch_size,
        "learning rate" : learning_rate,
        "hidden_dim"    : osc_hidden_size,
        "depth"         : osc_depth,
        "num workers"   : num_workers
    }
)

cfg = wandb.config
run_name = wandb.run.name 

for epoch in tqdm(range(cfg.epochs), desc="Training"):

    osc_deepONet.train()
    train_loss = 0.0

    