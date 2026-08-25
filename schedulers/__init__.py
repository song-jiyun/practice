import torch

from .step import load_steplr
from .cosine import load_cosinelr
from .plateau import load_plateaulr

scheduler_list = {
    "StepLR": load_steplr,
    "CosineAnnealingLR": load_cosinelr,
    "ReduceLROnPlateau": load_plateau,    
}

def get_scheduler_list():
    return list(scheduler_list.keys())

def load_scheduler(config, optimizer):
    name = config["scheduler"]

    scheduler = scheduler_list[name](config, optimizer)

    return scheduler