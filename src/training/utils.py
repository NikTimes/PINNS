import torch
from torch.utils.data import DataLoader

# ------------------------------------------------
# Dataloader Helper Function
# ------------------------------------------------

def build_dataloaders(train_dataset, val_dataset, cfg):
    
    train_loader = DataLoader(train_dataset, 
                              batch_size     = cfg["batch_size"],
                              num_workers    = cfg["num_workers"])

    val_loader   = DataLoader(val_dataset,   
                              batch_size     = cfg["batch_size"],
                              num_workers    = cfg["num_workers"])

    return train_loader, val_loader