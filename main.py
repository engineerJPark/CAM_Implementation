import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train import train, PolyOptimizer

from cam import resnet_cam
from voc.voc import myVOCDetection, train_dl, trainval_dl, val_dl, classes
from eval_cam import eval_cam

if __name__ == '__main__': 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device('cpu')
        torch.manual_seed(42)
    print("running is done on : ", device)

    model = resnet_cam().to(device)

    # # model checkpoint reloading
    # PATH = './checkpoint/model_12_28_16_7_6'
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # model training
    lr = 0.01
    weight_decay = 0.0001
    # epochs = 10
    epochs = 1
    param_groups = model.trainable_parameters()
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': lr, 'weight_decay': weight_decay},
        {'params': param_groups[1], 'lr': 10*lr, 'weight_decay': weight_decay},],
        lr=lr, weight_decay=weight_decay, max_step=(len(train_dl) * epochs))
    criterion = nn.MultiLabelSoftMarginLoss().to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.1)
    loss_history = train(model, optimizer, criterion, train_dl, trainval_dl, val_chk_freq=1, epochs=epochs, scheduler=None, device=device)
    eval_cam(model)
    
    loss_history['train'] = loss_history['train'].numpy()
    loss_history['val'] = loss_history['val'].numpy()
    plt.plot(loss_history['train'])
    plt.plot(loss_history['val'])
    plt.savefig('./loss_history.png')
    np.savetxt('./loss_history_train.txt', loss_history['train'])
    np.savetxt('./loss_history_val.txt', loss_history['val'])
    
    ''' 
    # Getting CAM on validation & Get iou for CAM
    model.switch2cam()
    model.eval()
    
    for i in range(2):
        val_img, val_label = next(iter(val_dl))
        out = model(normalization(val_img).to(device)) # 8,20,480,480
        val_img_pil = []
        for batch_idx in range(out.shape[0]):
            for channel_idx in range(out.shape[1]): # cam img for each class + coloring
                val_img_pil.append(PIL.Image.fromarray(np.uint8(cm.jet(out[batch_idx,channel_idx,:,:].detach().cpu().numpy()) * 255)))

        for batch_idx in range(out.shape[0]):
            for channel_idx in range(out.shape[1]): # superpose on image
                plt.imshow(val_img[batch_idx].detach().cpu().numpy().transpose(1, 2, 0), alpha = 0.4)
                plt.imshow(val_img_pil[8*batch_idx + channel_idx], alpha = 0.4)
                plt.savefig('./result/CAM_Result_%d_%s.png'%(8*i + batch_idx+1, classes[channel_idx]))
                plt.clf()
    '''
    