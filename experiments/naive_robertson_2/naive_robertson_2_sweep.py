# ============================================================
# Imports
# ============================================================

import sys
sys.path.insert(0, "../..")

from src.models        import build_logdeeponet
from src.data          import ODEIterableDataset, DirichletSampler, ConstrainedLHCSampler
from src.physics       import Robertson
from src.training      import Sweeper
from src.losses        import ScaledMSE

import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Sweep Config
# ============================================================

naive_robertson_2_sweep = {
    "method": "bayes",
    
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    
    "parameters": {
        # Tunable hyperparameters
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2
        },
        "batch_size": {
            "values": [16, 32, 64]
        },
        "hidden_size": {
            "values": [32, 64, 128]
        },
        "depth": {
            "values": [4, 6, 8]
        },
        "latent_size": {
            "values": [60, 90, 120]
        },
        "activation": {
            "values": ["tanh", "gelu", "sigmoid"]
        },

        # Fixed parameters
        "num_epochs":   {"value": 250},
        "input_size_b": {"value": 3},
        "input_size_t": {"value": 1},
        "output_size":  {"value": 3},
        "num_workers":  {"value": 12},
        "Save_model":   {"value": False},
        "t_span":       {"value": (1e-4, 1e6)}
    }
}

# ============================================================
# Physics System Class Initialization
# ============================================================

# Initialize Roberston Model
k1 = 4e-2
k2 = 3e7
k3 = 1e4
system = Robertson([k1, k2, k3])

# Initialize Sampler Object 
sampler = ConstrainedLHCSampler(low=0.5)

# Define the t_span 
t_span = (1e-4, 1e6)

# define Scaling for loss function 
y_scale = [1.0, 3.6e-5, 1.0]

# ============================================================
# Sweep Inputs
# ============================================================

# Initialize Training and validation datasets
train_dataset    = ODEIterableDataset(size = 2000,
                                      system_class = system,
                                      sampler      = sampler,
                                      t_span       = t_span,
                                      method       = "BDF",
                                      log_sampling = True,
                                      output_mask  = None)

val_dataset      = ODEIterableDataset(size = 100,
                                      system_class = system,
                                      sampler      = sampler,
                                      t_span       = t_span,
                                      method       = "BDF", 
                                      log_sampling = True,
                                      output_mask  = None)

model_builder = build_logdeeponet
optimizer = torch.optim.Adam
loss_fn   = ScaledMSE(y_scale).to(device)

# ============================================================
# Sweep 
# ============================================================

if __name__ == "__main__":
    osc_sweeper = Sweeper(
        model_builder = model_builder,
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        sweep_config  = naive_robertson_2_sweep,
        loss_fn       = loss_fn,
        optimizer     = torch.optim.Adam,
        project       = "robertson-2-sweep"
    )
    osc_sweeper.run(count=20)