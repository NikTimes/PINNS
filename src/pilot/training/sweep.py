import wandb
import torch.optim as optim
from .trainer import Trainer


class Sweeper():

    def __init__(self, model_builder, train_dataset, val_dataset, sweep_config, loss_fn, optimizer, project="PINNS-testing"):

        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.loss_fn       = loss_fn
        self.sweep_config  = sweep_config
        self.project       = project
        self.optimizer     = optimizer
        self.model_builder = model_builder

    def sweep_run(self):
        
        # Jupyter Notebook Check
        wandb.init()
        cfg = dict(wandb.config)

        # Build Model from Sweep config 
        model   = self.model_builder(cfg)

        # Train Model from config 
        trainer = Trainer(
            model         = model,
            train_dataset = self.train_dataset,
            val_dataset   = self.val_dataset,
            optimizer     = self.optimizer,
            loss_fn       = self.loss_fn,
            train_cfg     = {},
            model_cfg     = {}
        )

        trainer.run(config=cfg)
        wandb.finish()

    def run(self, count=10):

        sweep_id = wandb.sweep(self.sweep_config, project=self.project)
        wandb.agent(sweep_id, function=self.sweep_run, count=count)    

