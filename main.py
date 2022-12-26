from matplotlib import cm
import matplotlib.pyplot as plt
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train import train

from cam import resnet_cam
from voc.voc import myVOCDetection, train_dl, trainval_dl, val_dl

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
    
    # lr = 1e-3
    # weight_decay = 5e-4
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    train(model, optimizer, criterion, train_dl, trainval_dl, scheduler=None, epochs=10, device=device)

    model.switch2cam()
    for iter, (val_img, _) in enumerate(val_dl): # _ was target bb & labels, val_img, _ = next(iter(val_dl))
        out = model(val_img.to(device))
        val_img_pil = torch.zeros((out.shape[1], out.shape[2], out.shape[3]))
        for channel_idx in range(out.shape[1]): # cam img for each class + coloring
            val_img_pil[channel_idx,:,:] = PIL.Image.fromarray(np.uint8(cm.jet(out[0,channel_idx,:,:].detach().cpu().numpy()) * 255))

        # superpose on image
        val_img = val_img[0].detach().cpu().numpy().transpose(1, 2, 0)

        for channel_idx in range(out.shape[1]): 
            plt.imshow(val_img, alpha = 0.5)
            plt.imshow(val_img_pil[channel_idx], alpha = 0.3)
            plt.show()
            plt.savefig('./result/CAM_Result_%d_%d.png'%(iter+1, channel_idx+1))