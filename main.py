import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train import train, PolyOptimizer

from cam import Net, CAM
from voc.voc import myVOCDetection, train_dl, trainval_dl, val_dl, classes
from eval_cam import eval_cam
from print_cam import print_cam

if __name__ == '__main__': 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device('cpu')
        torch.manual_seed(42)
    print("running is done on : ", device)


    ############################ model training ############################
    model = Net().to(device)

    ### model checkpoint reloading
    PATH = './checkpoint/model_1_3_1_35_10'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    lr = 0.01
    weight_decay = 1e-5
    epochs = 10
    # epochs = 1
    
    param_groups = model.trainable_parameters()
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': lr, 'weight_decay': weight_decay},
        {'params': param_groups[1], 'lr': 10*lr, 'weight_decay': weight_decay},],
        lr=lr, weight_decay=weight_decay, max_step=(len(train_dl) * epochs))
    criterion = nn.MultiLabelSoftMarginLoss().to(device)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.1)
    scheduler = None
    loss_history, PATH = train(model, optimizer, criterion, train_dl, trainval_dl, save_freq=1, epochs=epochs, scheduler=None, device=device)
    
    loss_history['train'] = loss_history['train']
    loss_history['val'] = loss_history['val']
    plt.plot(loss_history['train'], label='train loss')
    plt.plot(loss_history['val'], label='validation loss')
    plt.legend()
    plt.savefig('./loss_history.png')
    plt.clf()
    
    with open('./loss_history.txt','w',encoding='UTF-8') as f:
        for i in range(len(loss_history['val'])):
            f.write('train loss : ' + str(loss_history['train'][i]) + ', val loss : ' + str(loss_history['val'][i]) + '\n')
            
    ############################ CAM print & evaluation ############################
    cam_model = CAM().to(device) # model checkpoint reloading
    # PATH = './checkpoint/model_12_29_18_28_5'
    cam_model.load_state_dict(torch.load(PATH)['model_state_dict'], strict=True)

    cam_metric = eval_cam(cam_model, device=device)
    print_cam(cam_model, device=device)