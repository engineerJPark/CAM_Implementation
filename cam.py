from torchvision.models import resnet101, ResNet101_Weights
import torch.nn as nn

resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

print(resnet)

class resnet_cam:
  def __init__(self):
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu
    self.maxpool = resnet.maxpool
    self.layer1 = resnet.layer1
    self.layer2 = resnet.layer2
    self.layer3 = resnet.layer3
    self.layer3 = resnet.layer3
    self.layer4 = resnet.layer4
    self.fc = resnet.fc
    self.avgpool = resnet.avgpool # gonna kill this 
    
    self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
    
  def forward(self, x):
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
        x[i,j] = self.fc(x[i,j])
        
        
    
    # need upsampling to the input size : bilinear 8 times
    out = x
    return out
  
# # train, validation dataset generating
# train_ds = myVOCDetection(path2data, year='2012', image_set='train', download=True)
# val_ds = myVOCDetection(path2data, year='2012', image_set='test', download=True)

# # dataloader generating
# train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
# val_dl = DataLoader(val_ds, batch_size=4, shuffle=True)

# coloring + superpose on image
