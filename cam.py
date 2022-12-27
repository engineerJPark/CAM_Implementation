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
            print(x.shape)
            x = F.softmax(x, dim=1)
            out = F.upsample(x, size=(600,600), mode='bilinear')
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


        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=self.n_classes, bias=True),
        #     nn.Softmax(dim=-1)
        # )

            # # 이부분 큰 수정이 필요하다
            # out = torch.zeros([1, self.n_classes, 19,19])
            # # pass fc for every spatial position
            # for i in range(x.shape[-2]):
            #     for j in range(x.shape[-1]):
            #         out[:,:,i,j] = self.fc(x[:,:,i,j]) # only 1 batch
            # out = self.upsampling(out)
            # out = (out - torch.unsqueeze(torch.min(out, dim=1)[0], 0)) * 255
            # out = out.to(torch.int)


        # self.softmax = nn.Softmax(dim=1)

        # self.upsampling = nn.UpsamplingBilinear2d(size=(600,600))