import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeModel(nn.Module):

    def __init__(self):
        super().__init__()

        base = models.resnet18(pretrained=True)

        self.features = nn.Sequential(*list(base.children())[:-1])

        self.classifier = nn.Linear(512,1)

    def forward(self,x):

        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)

        return x