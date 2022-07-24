import torch
from hydra.utils import to_absolute_path
from torch import nn
import torchvision

import alexnet


class EfficientNetB0Encoder(nn.Module):
    def __init__(self):
        super(EfficientNetB0Encoder, self).__init__()
        self.encoder = torchvision.models.efficientnet_b0(pretrained=True)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        x = self.encoder.features(image)
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class AlexNet224(nn.Module):
    def __init__(self, hidden_dim):
        super(AlexNet224, self).__init__()

        self.alex_net = alexnet.MyAlexNetCMC()
        self.alex_net.load_state_dict(torch.load(to_absolute_path('tmp/CMC_alexnet.pth'))['model'])

        self.fc_l = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),
            nn.Sigmoid()
        )

        self.fc_ab = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_l, x_ab = self.alex_net(x)
        x_l = self.fc_l(x_l)
        x_ab = self.fc_ab(x_ab)
        return x_l, x_ab
