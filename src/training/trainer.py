from .utils import build_dataloaders
from .engine import train_one_epoch, validate

import torch
from torch.utils.data import DataLoader
import numpy as np 
import wandb
from tqdm import tqdm 
import os





class Trainer():

    def __init__(self, model, train_dataset, val_dataset, optimizer, loss_fn, train_cfg, model_cfg):

        self.model         = model
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.loss_fn       = loss_fn
        self.optimizer     = optimizer
        self.train_cfg     = train_cfg
        self.model_cfg     = model_cfg
    
    def run(self, config=None):

        # Detect GPU 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check for Sweep Configuration
        if not config:
            config = {**self.train_cfg, **self.model_cfg, "device": str(device)}

        # Initialize Weights and Biases
        wandb.init(
            project = "PINNS-testing",
            config  = config,
            mode    = "online"
        )

        cfg = wandb.config

        # Initialize training and validation dataloaders 
        train_loader, val_loader = build_dataloaders(self.train_dataset, self.val_dataset, cfg)
        
        # Set model to device
        self.model = self.model.to(device)

        # Configure Optimizer
        self.optimizer = self.optimizer(self.model.parameters(), lr = cfg["learning_rate"])

        for epoch in tqdm(range(cfg['num_epochs']), desc="Training"):
            
            # Train epoch
            train_loss = train_one_epoch(self.model, train_loader, self.optimizer, self.loss_fn, device)

            # Validation Epoch
            val_loss   = validate(self.model, train_loader, self.loss_fn, device)

            # Weights And Biases Log 
            wandb.log({"train_loss": np.log(train_loss), "val_loss": np.log(val_loss)})
        
        # Save the Model weights 
        if cfg["Save_model"]: 
            abs_path = os.path.abspath(cfg["Save_directory"])
            torch.save(self.model.state_dict(), abs_path)
            wandb.save(abs_path)


"""
TRAIN_CONFIG = {

    "num_epochs"     : 100,
    "learning_rate"  : 0.0068741,
    "batch_size"     : 32,
    "num_workers"    : 2,
    "Save_model"     : False,
    "Save_directory" : None 

}


DEEPONET_CONFIG = {
    
    "hidden_size" : 32,
    "depth"       : 4,
    "latent_size" : 50,
    "input_size_b": 2,
    "input_size_t": 1,
    "output_size" : 1,
    "activation"  : "tanh",

}
"""