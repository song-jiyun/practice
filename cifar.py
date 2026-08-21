import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import SGD
import torchvision.models as models

import training as tr

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}, torch version: {torch.__version__}")

BATCH_SIZE = 64
EPOCHS = 300
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

class WideBasic(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else None
        self.equal_in_out = (in_planes == out_planes)
        self.shortcut = nn.Sequential()
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = x if self.equal_in_out else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'Depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasic, n_stages[0], n_stages[1], n, stride=1, drop_rate=dropout_rate)
        self.layer2 = self._wide_layer(WideBasic, n_stages[1], n_stages[2], n, stride=2, drop_rate=dropout_rate)
        self.layer3 = self._wide_layer(WideBasic, n_stages[2], n_stages[3], n, stride=2, drop_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(n_stages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_stages[3], num_classes)

        self._initialize_weights()

    def _wide_layer(self, block, in_planes, out_planes, num_blocks, stride, drop_rate):
        layers = []
        for i in range(num_blocks):
            current_stride = stride if i == 0 else 1
            input_planes = in_planes if i == 0 else out_planes
            layers.append(block(input_planes, out_planes, current_stride, drop_rate))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)

model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

tr.training(EPOCHS, model, train_loader, test_loader, optimizer, criterion, scheduler, name='cifar100', device=device, cutmix_prob=CUTMIX_PROB, cutmix_alpha=CUTMIX_ALPHA)