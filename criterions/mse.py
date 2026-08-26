import torch.nn as nn
import torch.nn.functional as F


class MSEForClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, target):
        target = F.one_hot(
            target,
            num_classes=self.num_classes
        ).float()

        probs = F.softmax(logits, dim=1)

        return nn.MSELoss()(probs, target)

def load_mseloss(config, info):
    return MSEForClassification(info["num_classes"])