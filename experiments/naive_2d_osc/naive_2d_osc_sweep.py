# ============================================================
# Imports
# ============================================================

import sys
sys.path.insert(0, "../..")

from src.models        import build_deeponet
from src.data          import ODEIterableDataset, LatinHypercubeSampler
from src.physics       import harm_osc
from src.training      import Sweeper

import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Sweep Config
# ============================================================

naive_2d_osc_sweep_config = {

    "method": "bayes",
    
    "metric": {"name": "val_loss", "goal": "minimize"},
    
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size":    {"values": [16, 32, 64]},
        "hidden_size":   {"values": [32, 64, 128]},
        "depth":         {"values": [2, 4, 6]},
        "latent_size":   {"values": [32, 50, 100]},
        "activation":    {"values": ["tanh", "relu", "sigmoid"]},
        
        
        # Fixed
        "num_epochs":    {"value": 80},
        "input_size_b":  {"value": 2},
        "input_size_t":  {"value": 1},
        "output_size":   {"value": 2},
        "num_workers":   {"value": 2},
        "Save_model":    {"value": False},
    }
}


# ============================================================
# Physics System Class Initialization
# ============================================================


# Initialize Harmonic Oscillator Object 
k = 2.0
c = 0.1
system = harm_osc([k, c])

# Initialize Sampler Object 
sampler = LatinHypercubeSampler(

    dimensions=2,
    lows      = [-1.0, -1.0],
    highs     = [1.0, 1.0]
    
)

# ============================================================
# Sweep Inputs
# ============================================================


# Initialize Training and validation datasets

train_dataset    = ODEIterableDataset(size = 1000,
                                      system_class = system,
                                      sampler      = sampler,
                                      t_span       = (0, 10),
                                      output_mask  = None)

val_dataset      = ODEIterableDataset(size = 10,
                                      system_class = system,
                                      sampler      = sampler,
                                      t_span       = (0, 10),
                                      output_mask  = None)

model_builder = build_deeponet
optimizer = torch.optim.Adam
loss_fn   = torch.nn.MSELoss()

# ============================================================
# Sweep 
# ============================================================

if __name__ == "__main__":
    osc_sweeper = Sweeper(
        model_builder = model_builder,
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        sweep_config  = naive_2d_osc_sweep_config,
        loss_fn       = torch.nn.MSELoss(),
        optimizer     = torch.optim.Adam,
        project       = "PINNS-Testing"
    )
    osc_sweeper.run(count=20)