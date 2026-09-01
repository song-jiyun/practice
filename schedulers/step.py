import torch.optim.lr_scheduler as lr


def load_steplr(config, optimizer, epoch):
    scheduler = lr.StepLR(
        optimizer,
        step_size=config["step_size"],
        gamma=config["gamma"],
    )
    return scheduler
