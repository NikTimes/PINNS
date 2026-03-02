import wandb
from training_loop import train

# -------------------------------------------------------------------------------
# 1. Sweep Config
# -------------------------------------------------------------------------------

sweep_config = {
    "method": "random",

    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },

    "parameters": {
    "learning_rate": {
        "distribution": "log_uniform_values",
        "min": 1e-4,
        "max": 1e-2
        },
    "batch_size":         {"values": [16, 32, 64]},
    "hidden_size":        {"values": [32, 64, 128, 256]},
    "depth":              {"values": [2, 3, 4]},
    "train_dataset_size": {"values": [500, 1000]},
    "val_dataset_size":   {"value": 100} 
    }
}

# -------------------------------------------------------------------------------
# 2. Run Sweep
# -------------------------------------------------------------------------------

sweep_id = wandb.sweep(sweep_config, project="Harmonic oscillator sweep")
wandb.agent(sweep_id, function=train, count=20) 