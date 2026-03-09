from model import *
from utils import *
from dataset import ODEIterableDataset

import torch
from torch.utils.data import DataLoader

import numpy as np 
import wandb
from tqdm import tqdm 

# -------------------------------------------------------------------------------
# 0. Hyper Parameters 
# -------------------------------------------------------------------------------

# Detect GPU 

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save the model

Save_model    = False
# Training Parameters 

num_epochs  = 100
num_workers = 12

# Loss Functions 

loss_fn     = torch.nn.MSELoss()

# Fixed branch and trunk input/output sizes

osc_input_size_b  = 2
osc_input_size_t  = 1
osc_latent_size   = 50
osc_output_size   = 2

# DeepONet output suze

osc_output_size   = 2

# Harmonic Oscillator 

string_constant = 2
dampening_coeff = 0.1
args1           = [string_constant, dampening_coeff]

osc_sampler     = LatinHypercubeSampler(dimensions = 2, 
                                        lows       = [-1, -1], 
                                        highs      = [1, 1])

osc_object          = harm_osc(args1)
osc_t_span          = (0, 10)


# -------------------------------------------------------------------------------
# 1. Training Function called by WANDB agent
# -------------------------------------------------------------------------------

def train(config=None):

    # Initialize WANDB config
    wandb.init(config=config)
    wandb.config.update({"device": str(device)})
    cfg = wandb.config


    # Set config hyperparameters
    learning_rate      = cfg.learning_rate
    batch_size         = cfg.batch_size
    depth              = cfg.depth
    hidden_size        = cfg.hidden_size
    train_dataset_size = cfg.train_dataset_size
    val_dataset_size   = cfg.val_dataset_size

    # Initialize datasets 
    train_osc_dataset  = ODEIterableDataset(size         = train_dataset_size,
                                            system_class = osc_object, 
                                            sampler      = osc_sampler, 
                                            t_span       = osc_t_span,
                                            method       = "RK45")

    val_osc_dataset    = ODEIterableDataset(size         = val_dataset_size,
                                            system_class = osc_object, 
                                            sampler      = osc_sampler, 
                                            t_span       = osc_t_span,
                                            method       = "RK45")

    # Initialize dataloaders
    train_osc_loader = DataLoader(dataset     = train_osc_dataset, 
                                  batch_size  = batch_size,
                                  num_workers = num_workers)

    val_osc_loader   = DataLoader(dataset     = val_osc_dataset, 
                                  batch_size  = batch_size,
                                  num_workers = num_workers)
    
    # Initialize Branch and Trunk Nets 
    osc_branch_net   = General_MLP(input_size  = osc_input_size_b, 
                                   output_size = osc_output_size,
                                   depth       = depth, 
                                   hidden_size = hidden_size, 
                                   act         = nn.Tanh())

    osc_trunk_net    = General_MLP(input_size  = osc_input_size_t, 
                                   output_size = osc_output_size,
                                   depth       = depth, 
                                   hidden_size = hidden_size, 
                                   act         = nn.Tanh())

    # Initialize DeepONet 
    osc_deepONet     = DeepONet(branch_net  = osc_branch_net, 
                                trunk_net   = osc_trunk_net,
                                output_size = osc_output_size).to(device=device)
    
    # Initialize Adam Optimizer
    optimizer        = torch.optim.Adam(osc_deepONet.parameters(), 
                                        lr=learning_rate)


    # -------------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------------

    for epoch in tqdm(range(num_epochs), desc="Training"):

        # Set Network to training mode 
        osc_deepONet.train()
        running_loss = 0.0
        num_steps    = 0.0

        # Training Steps
        for I, t, y in train_osc_loader:

            # Use GPU, also note y[:, 0:1] because we are only extracting first output of solve_ivp
            I, t, y = I.to(device), t.to(device), y.to(device)
            
            # Network forward Pass
            pred = osc_deepONet(I, t)

            # Loss calculation
            loss = loss_fn(pred, y)

            # Optimizer Steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_steps  += 1
        
        train_loss = running_loss / num_steps # this can probably be changed to len(dataloader)

        # Validation Steps

        # Set model to evaluation mode
        osc_deepONet.eval()
        running_loss   = 0.0
        num_steps      = 0.0

        # Validation Steps 
        with torch.no_grad():
            for I, t, y in val_osc_loader:
                
                I, t, y = I.to(device), t.to(device), y.to(device)

                # Network Forward Pass
                pred = osc_deepONet(I, t)

                # Loss calculation
                loss = loss_fn(pred, y)

                running_loss += loss.item()
                num_steps += 1

        val_loss = running_loss / num_steps

        # Weights And Biases Log 
        wandb.log({"train_loss": np.log(train_loss), "val_loss": np.log(val_loss)})


    if Save_model:
        torch.save(osc_deepONet.state_dict(), "weights/best_model.pth")
        wandb.save("weights/best_model.pth")  # syncs the file to W&B




"""
train(config={
    "learning_rate"     : 0.0068741,
    "batch_size"        : 32,
    "hidden_size"       : 32,
    "depth"             : 4,
    "train_dataset_size": 1000,
    "val_dataset_size"  : 100,
})
"""