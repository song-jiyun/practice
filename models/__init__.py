from torchvision import models
class Net():
    pass

model_list = {
    "CUSTOM": Net,
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50,
    "ViT-Base/16": models.vit_b_16,
    "ViT-Base/32": models.vit_b_32,
    "ViT-Large/16": models.vit_l_16,
    "ViT-Large/32": models.vit_l_32,
    "ViT-Huge/14": models.vit_h_14,
}

def get_model_list():
    return list(model_list.keys())

def load_model():
    pass