from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn

resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

# print(resnet)

class resnet_cam(nn.Module):
    def __init__(self, n_classes=20):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        # self.fc = resnet.fc # ImageNet pretrained has 1000 classes
        self.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        self.upsampling = nn.UpsamplingBilinear2d(size=(600,600))

        self.cam_flag = False


    def forward(self, x):
        if self.cam_flag == True:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            out = torch.zeros([1,n_classes,19,19])
            
            # pass fc for every spatial position
            for i in range(x.shape[-2]):
                for j in range(x.shape[-1]):
                    out[:,:,i,j] = self.fc(x[:,:,i,j]) # only 1 batch
            out = self.upsampling(out)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            out = self.fc(x.reshape(-1))
        return out

    def switch2forward(self):
        self.cam_flag = False
        print("forward mode for train")

    def switch2cam(self):
        self.cam_flag = True
        print("CAM mode")


# test_model = resnet_cam()
# for param in test_model.parameters():
#     param.requires_grad = False
# test_model.fc.weight.requires_grad = True
# test_model.fc.bias.requires_grad = True
