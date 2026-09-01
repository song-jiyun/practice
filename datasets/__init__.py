from torch.utils.data import DataLoader

from .cifar import load_cifar10, load_cifar100
from .mnist import load_mnist
from .imagenet import load_imagenet

dataset_list = {
    "MNIST": load_mnist,
    "CIFAR10": load_cifar10,
    "CIFAR100": load_cifar100,
    "ImageNet": load_imagenet,
}

def get_dataset_list():
    return list(dataset_list.keys())


def load_dataset(config):
    name = config["dataset"]
    batch_size = config["batch_size"]

    train_dataset, test_dataset, info = dataset_list[name]()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader, info
