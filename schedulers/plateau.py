import torch.optim.lr_scheduler as lr


def load_plateaulr(config, optimizer, epoch):
    scheduler = lr.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["factor"],
        patience=config["patience"],
        min_lr=config["min_lr"],
    )
    return scheduler
