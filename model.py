import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):

    def __init__(self):
        super(SimCLR, self).__init__()
        self.resnet = models.resnet18(pretrained=False, num_classes = 2048)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))
    
    def forward(self, x):
        x = self.resnet(x)
        return self.mlp(z)

