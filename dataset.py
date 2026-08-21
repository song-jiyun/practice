from torchvision import datasets

dataset_list = {
    "MNIST": datasets.MNIST,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
    "ImageNet": datasets.ImageNet,
}

def get_dataset_list():
    return list(dataset_list.keys())

def load_dataset():
    pass
