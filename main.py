import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train import train, PolyOptimizer

from cam import resnet_cam
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

    model = resnet_cam().to(device)

    # # model checkpoint reloading
    # PATH = './checkpoint/model_12_28_22_24_1'
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # model training
    lr = 0.001
    weight_decay = 0.0001
    epochs = 30
    # epochs = 1
    
    param_groups = model.trainable_parameters()
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': lr, 'weight_decay': weight_decay},
        {'params': param_groups[1], 'lr': 10*lr, 'weight_decay': weight_decay},],
        lr=lr, weight_decay=weight_decay, max_step=(len(train_dl) * epochs))
    criterion = nn.MultiLabelSoftMarginLoss().to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.1)
    loss_history, PATH = train(model, optimizer, criterion, train_dl, trainval_dl, save_freq=1, epochs=epochs, scheduler=None, device=device)
    
    # model checkpoint reloading
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    eval_cam(model, device=device)
    print_cam(model, device=device)
    
    loss_history['train'] = loss_history['train']
    loss_history['val'] = loss_history['val']
    plt.plot(loss_history['train'])
    plt.plot(loss_history['val'])
    plt.savefig('./loss_history.png')
    
    with open('./loss_history.txt','w',encoding='UTF-8') as f:
        for i in range(len(loss_history['val'])):
            f.write('train loss : ' + str(loss_history['train'][i]) + ', val loss : ' + str(loss_history['val'][i]))
    