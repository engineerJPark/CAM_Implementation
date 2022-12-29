from matplotlib import cm
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from train import normalization
from voc.voc import classes
from chainercv.datasets import VOCSemanticSegmentationDataset

def print_cam(model, device='cpu'):
    model.switch2cam()
    
    dataset = VOCSemanticSegmentationDataset(split='train', data_dir='./voc/VOCdevkit/VOC2012')
    img = [dataset.get_example_by_keys(i, (0,))[0] for i in range(len(dataset))] 
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))] # segmentation label for all sample
    
    preds = []
    for i in range(len(dataset)):
        val_img = torch.from_numpy(img[i]).unsqueeze(dim=0) / 255.
        keys = np.unique(labels[i])[2:]
        cams = model(normalization(val_img).to(device)).squeeze()[keys - 1].detach().cpu().numpy() # 1,20,480,480 -> n_of_GT,480,480
              
        # for CAM image
        val_img_pil = []
        for channel_idx in range(cams.shape[0]): # cam img for each class + coloring
            val_img_pil.append(PIL.Image.fromarray(np.uint8(cm.jet(cams[channel_idx,:,:]) * 255)))

        for channel_idx in range(cams.shape[0]): # superpose on image
            plt.imshow(val_img.squeeze().detach().cpu().numpy().transpose(1, 2, 0), alpha = 0.4)
            plt.imshow(val_img_pil[channel_idx], alpha = 0.4)
            plt.savefig('./result/CAM_Result_%d_%s.png' % (i, classes[keys[channel_idx] - 1]))
            plt.clf()
