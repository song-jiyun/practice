from .mse import load_mseloss
from .ce import load_celoss

criterion_list = {
    "CrossEntropyLoss": load_celoss,
    "MSELoss": load_mseloss,
}

def get_criterion_list():
    return list(criterion_list.keys())


def load_criterion(config, info):
    name = config["criterion"]
    criterion = criterion_list[name](config, info)

    return criterion
