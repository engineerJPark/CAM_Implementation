from matplotlib import cm
import matplotlib.pyplot as plt
import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader

from cam import resnet_cam
from voc.voc import myVOCDetection, val_ds, val_dl

if __name__ == '__main__': 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device('cpu')
        torch.manual_seed(42)
    print(torch.__version__, device)

    model = resnet_cam().to(device)

    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    
    idx = 0 # for iter, (val_img, _, _) in enumerate(val_dl): # _ was target bb & labels
    val_img, _, _ = next(iter(val_dl)) 
    # print('val_img.shape : ', val_img.shape)
    out = model(val_img.to(device))

    print(out.shape)
    print(out[0,0,:,:].shape)

    # cam img for each class + coloring
    # for channel_idx in range(out.shape[0]):
    val_img_pil = PIL.Image.fromarray(np.uint8(cm.jet(out[0,0,:,:].detach().cpu().numpy() * 1) * 255)) # error check, 10, 255

    # superpose on image
    # test on class 0
    val_img = val_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    plt.imshow(val_img, alpha = 0.5)
    plt.imshow(val_img_pil, alpha = 0.5)
    plt.show()
    plt.savefig('cam01.png')


# 저게 1000개니까 fc 교체하고 다시 train 해야하지 않나?