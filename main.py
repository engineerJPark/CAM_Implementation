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
    optimizer = optim.Adam(model.parameters(), lr=lr, eight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    train(model, optimizer, criterion, train_dl, trainval_dl, 
            scheduler=None, epochs=100, device=device)

    model.switch2cam()
    idx = 0 # for iter, (val_img, _, _) in enumerate(val_dl): # _ was target bb & labels
    val_img, _, _ = next(iter(val_dl)) 
    out = model(val_img.to(device))

    # cam img for each class + coloring
    # for channel_idx in range(out.shape[0]):
    val_img_pil = PIL.Image.fromarray(np.uint8(cm.jet(out[0,6,:,:].detach().cpu().numpy() * 1) * 255)) # 6 : car class

    # superpose on image, test on class 0
    val_img = val_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    plt.imshow(val_img, alpha = 0.5)
    plt.imshow(val_img_pil, alpha = 0.3)
    plt.show()
    plt.savefig('CAM_Result01.png')