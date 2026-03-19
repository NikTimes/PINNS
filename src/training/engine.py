import numpy as np 

# ------------------------------------------------
# Helper Training Functions
# ------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
        
    # Set Network to training mode 
    model.train()
    total_loss  = 0.0
    steps       = 0

    # Training Steps
    for batch in loader:

        inputs  = [x.to(device) for x in batch[:-1]]  # all but last = inputs
        targets = batch[-1].to(device)
        
        # Network forward Pass
        pred = model(*inputs) # Unpack inputs

        # Loss calculation
        loss = loss_fn(pred, targets)

        # Optimizer Steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps      += 1
    
    return total_loss / steps

def validate(model, loader, loss_fn, device):

    model.eval()
    total_loss  = 0.0
    steps       = 0

    for batch in loader:

        inputs  = [x.to(device) for x in batch[:-1]]  # all but last = inputs
        targets = batch[-1].to(device)
        
        # Network forward Pass
        pred = model(*inputs) # Unpack Inputs

        # Loss calculation
        loss = loss_fn(pred, targets)

        # Calculate Loss terms 
        total_loss   += loss.item()
        steps        += 1
    
    return total_loss / steps