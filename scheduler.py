import torch

def build_cosine(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def build_step(optimizer, epochs):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

scheduler_list = {
    "CosineAnnealingLR": build_cosine,
    "StepLR": build_step,
}

def get_scheduler_list():
    return list(scheduler_list.keys())

def load_scheduler(name, optimizer, epochs):
    builder = scheduler_list[name]
    return builder(optimizer, epochs)