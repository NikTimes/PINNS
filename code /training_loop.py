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

# WANDB

WAND = True 
# Training Parameters 

dataset_size          = 1000
num_epochs            = 20

batch_size            = 32
num_workers           = 4
learning_rate         = 0.032

# Loss Functions 

loss_fn     = torch.nn.MSELoss()

# Harmonic Oscillator 

string_constant = 2
dampening_coeff = 0.1
args1           = [string_constant, dampening_coeff]

osc_sampler     = LatinHypercubeSampler(dimensions = 2, 
                                        lows       = [-1, -1], 
                                        highs      = [1, 1])

osc_object          = harm_osc(args1)
osc_dataset_samples = dataset_size
osc_t_span          = (0, 100)

osc_dataset     = ODEIterableDataset(size         = dataset_size,
                                     system_class = osc_object, 
                                     sampler      = osc_sampler, 
                                     t_span       = osc_t_span,
                                     method       = "RK45")

# -------------------------------------------------------------------------------
# 1. Retrieving data 
# -------------------------------------------------------------------------------

# Harmonic Oscillator 

osc_loader = DataLoader(dataset     = osc_dataset, 
                        batch_size  = batch_size,
                        num_workers = num_workers)


# -------------------------------------------------------------------------------
# 2. Models
# -------------------------------------------------------------------------------

# Harmonic Oscillator 

# Define branch and trunk network parameters 
osc_input_size_b  = 2
osc_input_size_t  = 1
osc_output_size   = 2
osc_depth         = 2
osc_hidden_size   = 32

# Initialize Branch and Trunk Nets 
osc_branch_net = General_MLP(input_size  = osc_input_size_b, 
                             output_size = osc_output_size,
                             depth       = osc_depth, 
                             hidden_size = osc_hidden_size, 
                             act         = nn.Tanh())

osc_trunk_net  = General_MLP(input_size  = osc_input_size_t, 
                             output_size = osc_output_size,
                             depth       = osc_depth, 
                             hidden_size = osc_hidden_size, 
                             act         = nn.Tanh())

# Initialize DeepONet 
osc_deepONet   = DeepONet(branch_net = osc_branch_net, 
                          trunk_net  = osc_trunk_net)


# Initialize Adam Optimizer
optimizer      = torch.optim.Adam(osc_deepONet.parameters(), 
                                  lr=learning_rate)

# -------------------------------------------------------------------------------
# 3. WandB config
# -------------------------------------------------------------------------------

if WAND: 
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

    cfg      = wandb.config
    run_name = wandb.run.name 
else:
    pass

# -------------------------------------------------------------------------------
# 4. Training Loop
# -------------------------------------------------------------------------------

for epoch in tqdm(range(num_epochs), desc="Training"):

    # Set Network to training mode 
    osc_deepONet.train()
    train_loss = 0.0
    num_steps  = 0.0

    # Training Steps
    for I, t, y in osc_loader:
        
        # Network forward Pass
        pred = osc_deepONet(I, t)

        # Loss calculation
        loss = loss_fn(pred, y)

        # Optimizer Steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_steps      += 1
    
    train_loss /= num_steps

    # Validation Steps

    # Set model to evaluation mode
    osc_deepONet.eval()
    val_loss = 0.0
    num_steps  = 0.0

    # Validation Steps 
    with torch.no_grad():
        for I, t, y in osc_loader:

            # Network Forward Pass
            pred = osc_deepONet(I, t)

            # Loss calculation
            loss = loss_fn(pred, y)

            val_loss  += loss.item()
            num_steps += 1

    val_loss /= num_steps 

    # Weights And Biases Log 
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})