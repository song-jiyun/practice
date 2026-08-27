from torchvision import models
import torch.nn as nn

def load_resnet(model, info):
    num_classes = info["num_classes"]
    in_channels = info["in_channels"]
    image_size = info["image_size"]

    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def load_resnet18(pretrained, info):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18()
    return load_resnet(model, info)

def load_resnet34(pretrained, info):
    if pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model = models.resnet34()
    return load_resnet(model, info)

def load_resnet50(pretrained, info):
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50()
    return load_resnet(model, info)