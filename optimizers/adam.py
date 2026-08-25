import torch

default_config = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 5e-4,
}

def load_adam(config, model):
    default_config["lr"] = config["lr"]
    default_config["beta1"] = config["momentum"]
    default_config["weight_decay"] = config["weight_decay"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=default_config["lr"],
        betas=(default_config["beta1"], default_config["beta2"]),
        eps=default_config["eps"],
        weight_decay=default_config["weight_decay"],
    )

    return optimizer

def load_adamw(config, model):
    default_config["lr"] = config["lr"]
    default_config["beta1"] = config["momentum"]
    default_config["weight_decay"] = config["weight_decay"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=default_config["lr"],
        betas=(default_config["beta1"], default_config["beta2"]),
        eps=default_config["eps"],
        weight_decay=default_config["weight_decay"],
    )

    return optimizer
