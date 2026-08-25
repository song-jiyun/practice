import torch

default_config = {
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4
}

def load_sgd(config, model):
    default_config["lr"] = config["lr"]
    default_config["momentum"] = config["momentum"]
    default_config["weight_decay"] = config["weight_decay"]

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=default_config["lr"],
        momentum=default_config["momentum"],
        weight_decay=default_config["weight_decay"],
    )

    return optimizer
