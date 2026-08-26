import torch

from .step import load_steplr
from .cosine import load_cosinelr
from .plateau import load_plateaulr

scheduler_list = {
    "StepLR": {
        "loader": load_steplr,
        "config": {
            "scheduler": "StepLR",
            "step_size": 30,
            "gamma": 0.1,
        },
        "type": "epoch",
    },

    "CosineAnnealingLR": {
        "loader": load_cosinelr,
        "config": {
            "scheduler": "CosineAnnealingLR",
            "eta_min": 0.0,
        },
        "type": "epoch",
    },

    "ReduceLROnPlateau": {
        "loader": load_plateaulr,
        "config": {
            "scheduler": "ReduceLROnPlateau",
            "factor": 0.1,
            "patience": 10,
            "min_lr": 0.0,
        },
        "type": "metric",
    },
}

def get_scheduler_list():
    return list(scheduler_list.keys())

def get_scheduler_config(name):
    return scheduler_list[name]['config']

def load_scheduler(config, optimizer, epoch):
    name = config["scheduler"]

    scheduler = scheduler_list[name]['loader'](config, optimizer, epoch)

    return scheduler