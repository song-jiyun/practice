from .sgd import load_sgd
from .adam import load_adam, load_adamw

optimizer_list = {
    "SGD": load_sgd,
    "Adam": load_adam,
    "AdamW": load_adamw,
}

def get_optimizer_list():
    return list(optimizer_list.keys())

def load_optimizer(config, model):
    name = config["optimizer"]
    optimizer = optimizer_list[name](config, model)

    return optimizer
