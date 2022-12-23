from torchvision.models import resnet101, ResNet101_Weights
import torch.nn as nn

resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

# print(resnet)

class resnet_cam(nn.Module):
  def __init__(self):
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

    # self.fc = resnet.fc # ImageNet pretrained has 1000 classes
    self.fc = nn.Linear(in_features=2048, out_features=21, bias=True)
    self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
    
  def forward(self, x): # 8 times stride 2
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer3(x)
    x = self.layer4(x)
    
    # pass fc for every spatial position
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        x[:,:,i,j] = self.fc(x[:,:,i,j]) # only 1 batch

    x = upsampling(x)
    x = upsampling(x)
    x = upsampling(x)
    x = upsampling(x)
    x = upsampling(x)
    x = upsampling(x)
    x = upsampling(x)
    x = upsampling(x)

    # need upsampling to the input size : bilinear 8 times
    out = x
    return out

# test_model = resnet_cam()
# for param in test_model.parameters():
#     param.requires_grad = False
# test_model.fc.weight.requires_grad = True
# test_model.fc.bias.requires_grad = True