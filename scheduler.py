import torch

def build_cosine(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def build_step(optimizer, epochs):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

SCHEDULER_REGISTRY = {
    "CosineAnnealingLR": build_cosine,
    "StepLR": build_step,
}

def get_scheduler_list():
    return list(SCHEDULER_REGISTRY.keys())

def load_scheduler(name, optimizer, epochs):
    builder = SCHEDULER_REGISTRY[name]
    return builder(optimizer, epochs)