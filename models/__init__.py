from .resnet import load_resnet18, load_resnet34, load_resnet50
from .vit import load_vitb16, load_vitb32, load_vitl16, load_vitl32, load_vith14
from .custom import load_customnet

model_list = {
    "CUSTOM": load_customnet,
    "ResNet18": load_resnet18,
    "ResNet34": load_resnet34,
    "ResNet50": load_resnet50,
    "ViT-Base/16": load_vitb16,
    "ViT-Base/32": load_vitb32,
    "ViT-Large/16": load_vitl16,
    "ViT-Large/32": load_vitl32,
    "ViT-Huge/14": load_vith14,
}

def get_model_list():
    return list(model_list.keys())

def load_model(config, info):
    name = config["model"]
    pretrained = config["pretrained"]
    device = config["device"]

    model = model_list[name](pretrained, info)
    
    if model is None:
        return None
    else:
        model = model.to(device)

    return model