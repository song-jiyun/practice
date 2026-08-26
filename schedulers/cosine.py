import torch.optim.lr_scheduler as lr

def load_cosinelr(config, optimizer, epoch):
    scheduler = lr.CosineAnnealingLR(
        optimizer,
        T_max=epoch,
        eta_min=config["eta_min"],
    )
    return scheduler