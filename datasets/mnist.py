from torchvision import datasets, transforms

def load_mnist():
    mean = (0.1307)
    std = (0.3081)
    image_size = 28

    info = {
        "num_classes": 10,
        "in_channels": 1,
        "image_size": image_size
    }

    transform_train = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset, info