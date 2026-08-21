import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.nn.init as init

import training as tr

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}, torch version: {torch.__version__}")

BATCH_SIZE = 64
EPOCHS = 1550
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0

mnist_mean = 0.1307
mnist_std = 0.3081

transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize((mnist_mean,), (mnist_std,)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mnist_mean,), (mnist_std,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()

# Use ResNet18 adapted for single-channel input
def get_resnet18(num_classes=10):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = get_resnet18().to(device)
model.apply(weights_init)

# Use SGD with momentum and weight decay for best generalization
# Scale learning rate for smaller batch (base lr 0.1 @ batch 128 -> 0.05 @ batch 64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Cosine LR schedule across full epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

tr.training(EPOCHS, model, train_loader, test_loader, optimizer, criterion, scheduler=scheduler,
            name='mnist_resnet18', device=device, cutmix_prob=CUTMIX_PROB, cutmix_alpha=CUTMIX_ALPHA)