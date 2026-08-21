import torch.nn as nn

CRITERION_REGISTRY = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
}

def get_criterion_list():
    return list(CRITERION_REGISTRY.keys())

def load_criterion(name):
    criterion_cls = CRITERION_REGISTRY[name]
    return criterion_cls()