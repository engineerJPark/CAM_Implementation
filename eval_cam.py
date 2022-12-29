from matplotlib import cm
import matplotlib.pyplot as plt
import PIL

import torch
import numpy as np
import os
from train import normalization
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

def eval_cam(model, device='cpu'):
    model.eval()
    print("CAM evaluation")
    cam_eval_thres = 0.15
    
    dataset = VOCSemanticSegmentationDataset(split='train', data_dir='./voc/VOCdevkit/VOC2012')
    img = [dataset.get_example_by_keys(i, (0,))[0] for i in range(len(dataset))] 
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))] # segmentation label for all sample
    
    preds = []
    for i in range(len(dataset)):
        val_img = torch.from_numpy(np.array([img[i], np.flip(img[i], -1)])) / 255.
        keys = np.unique(labels[i])[2:]
             
        cams = model(normalization(val_img).to(device)).squeeze()[keys - 1].detach().cpu().numpy() # 1,20,480,480 -> n_of_GT,480,480
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cam_eval_thres)
        keys = np.pad(keys, (1, 0), mode='constant')
        
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())


    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print({'iou': iou, 'miou': np.nanmean(iou)})
