from torchvision import models
import torch.nn as nn

def load_vit(model, info):
    num_classes = info["num_classes"]
    in_channels = info["in_channels"]
    image_size = info["image_size"]

    if image_size != 224:
        return None

    conv_proj = model.conv_proj
    if conv_proj.in_channels != in_channels:
        model.conv_proj = nn.Conv2d(in_channels, conv_proj.out_channels, kernel_size=conv_proj.kernel_size, stride=conv_proj.stride, padding=conv_proj.padding, bias=conv_proj.bias is not None)

    if hasattr(model, "heads") and isinstance(model.heads, nn.Linear):
        model.heads = nn.Linear(model.heads.in_features, num_classes)
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)

    return model

def load_vitb16(pretrained, info):
    if pretrained:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    else:
        model = models.vit_b_16()
    return load_vit(model, info)

def load_vitb32(pretrained, info):
    if pretrained:
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
    else:
        model = models.vit_b_32()
    return load_vit(model, info)

def load_vitl16(pretrained, info):
    if pretrained:
        model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
    else:
        model = models.vit_l_16()
    return load_vit(model, info)

def load_vitl32(pretrained, info):
    if pretrained:
        model = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT)
    else:
        model = models.vit_l_32()
    return load_vit(model, info)

def load_vith14(pretrained, info):
    if pretrained:
        model = models.vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT)
    else:
        model = models.vit_h_14()
    return load_vit(model, info)