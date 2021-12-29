import torch
import sys

def save_checkpoint(n_iter, epoch, model, optimer, file_path):
    """params:
    epcoh: the current epoch
    n_iter: the current iter
    model: the model dict
    optimer: the optimizer dict
    """
    state = {}
    state["n_iter"] = n_iter
    state["epoch"] = epoch
    state["lr"] = optimer.param_groups[0]["lr"]
    state["state_dict"] = model.state_dict()
    state["optimizer"] = optimer.state_dict()
   
    torch.save(state, file_path)