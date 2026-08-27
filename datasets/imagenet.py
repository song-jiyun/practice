from torchvision import datasets, transforms

def load_imagenet():
    mean = (0.4850, 0.4560, 0.4060)
    std = (0.2290, 0.2240, 0.2250)
    image_size = 224

    info = {
        "num_classes": 1000,
        "in_channels": 3,
        "image_size": image_size
    }

    transform_train = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size)
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageNet(root='./data', split='train', transform=transform_train)
    test_dataset = datasets.ImageNet(root='./data', split='val', transform=transform_test)

    return train_dataset, test_dataset, info