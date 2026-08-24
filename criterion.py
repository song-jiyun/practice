import torch.nn as nn

criterion_list = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
}

def get_criterion_list():
    return list(criterion_list.keys())

def load_criterion(name):
    criterion_cls = criterion_list[name]
    return criterion_cls()