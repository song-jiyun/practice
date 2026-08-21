import torch

def build_sgd(model, lr, weight_decay=0.0):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

def build_adam(model, lr, weight_decay=0.0):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

OPTIMIZER_REGISTRY = {
    "SGD": build_sgd,
    "Adam": build_adam,
}

def get_optimizer_list():
    return list(OPTIMIZER_REGISTRY.keys())

def load_optimizer(name, model, lr, weight_decay=0.0):
    builder = OPTIMIZER_REGISTRY[name]
    return builder(model, lr, weight_decay=weight_decay)