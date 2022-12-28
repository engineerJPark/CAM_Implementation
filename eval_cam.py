from matplotlib import cm
import matplotlib.pyplot as plt
import PIL
import numpy as np
import os
from voc.voc import myVOCDetection, path2data, val_transforms, classes

def eval_cam(model):
    '''
    cam_img : numpy image, give only GT channel, dimension should be (N of GT labels, row, column)
    val_label : not one hot vector, need class number
    '''
    model.switch2cam()
    model.eval()
    
    dataset = myVOCDetection(path2data, year='2012', image_set='val')
    dataset.transforms = val_transforms
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    cam_eval_thres = 0.15
    preds = []
    for i, (val_img, val_label) in enumerate(dataloader):
        cams = model(normalization(val_img).to(device)) # 8,20,480,480 -> for 1 batch
        cams = cams.numpy()
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cam_eval_thres)
        
        val_label = (val_label == 1)
        label = val_label.nonzero()
        print('label : ', label) # for debug
        label = np.pad(label + 1, (1, 0), mode='constant')
        
        cls_labels = np.argmax(cams, axis=0) # channel dimension?
        cls_labels = label[cls_labels]
        preds.append(cls_labels.copy())
        
        val_img_pil = []
        for batch_idx in range(cams.shape[0]): # 1 batch
            for channel_idx in (label-1): # cam img for each class + coloring
                val_img_pil.append(PIL.Image.fromarray(np.uint8(cm.jet(out[batch_idx,channel_idx,:,:].detach().cpu().numpy()) * 255)))

        for batch_idx in range(cams.shape[0]): # 1 batch
            for channel_idx in range(len(val_img_pil)): # superpose on image
                plt.imshow(val_img[batch_idx].detach().cpu().numpy().transpose(1, 2, 0), alpha = 0.4)
                plt.imshow(val_img_pil[channel_idx], alpha = 0.4)
                plt.savefig('./result/CAM_Result_%d_%s.png'%(i, classes[label[channel_idx]-1]))
                plt.clf()

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print({'iou': iou, 'miou': np.nanmean(iou)})
    
    
########################
'''
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