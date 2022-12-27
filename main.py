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
    
    lr = 0.001
    weight_decay = 0.0005
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) # # optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss().to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.1)
    train(model, optimizer, criterion, train_dl, trainval_dl, val_chk_freq=1, epochs=1, scheduler=None, device=device) # edit to 100 after the test

    model.switch2cam()
    for iter, (ori_val_img, val_img, _) in enumerate(val_dl): # _ was target bb & labels, val_img, _ = next(iter(val_dl))
        out = model(val_img.to(device))
        # val_img_pil = np.zeros((out.shape[1], out.shape[2], out.shape[3]))
        val_img_pil = []
        for channel_idx in range(out.shape[1]): # cam img for each class + coloring
            val_img_pil.append(PIL.Image.fromarray(np.uint8(cm.jet(out[0,channel_idx,:,:].detach().cpu().numpy()) * 255)))
        ori_val_img = ori_val_img[0].detach().cpu().numpy().transpose(1, 2, 0)

        for channel_idx in range(out.shape[1]): # superpose on image
            plt.imshow(ori_val_img, alpha = 0.5)
            plt.imshow(val_img_pil[channel_idx], alpha = 0.3)
            plt.show()
            plt.savefig('./result/CAM_Result_%d_%d.png'%(iter+1, channel_idx+1))