from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F

resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

# print(resnet)

class resnet_cam(nn.Module):
    def __init__(self, n_classes=20):
        super().__init__()
        self.n_classes = n_classes

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.classifier = nn.Conv2d(512, 20, 1, bias=False)

        self.cam_flag = False


    def forward(self, x):
        if self.cam_flag == True:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.conv2d(x, self.classifier.weight)
            x = F.softmax(x, dim=1)
            out = F.interpolate(x, size=(480, 480), mode='bilinear')
        else:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = self.classifier(x)
            x = F.softmax(x, dim=1)
            out = x.view(-1)
        return out

    def switch2forward(self):
        self.cam_flag = False
        print("Forward mode")

    def switch2cam(self):
        self.cam_flag = True
        print("CAM mode")