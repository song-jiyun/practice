import torch.nn as nn

def load_celoss(config, info):
    return nn.CrossEntropyLoss(label_smoothing=0.1)