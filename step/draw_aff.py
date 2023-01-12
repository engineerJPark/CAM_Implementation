import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL

import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

cudnn.enabled = True

def _work(process_id, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    
    os.makedirs(args.aff_out_dir + "_on_img", exist_ok=True)

    with torch.no_grad(), cuda.device(process_id):
        for iter, pack in enumerate(data_loader):
            # load cam npy
            name_str = pack['name'][0]
            label = pack['label'][0] # one hot encoded
            valid_cat = torch.nonzero(label)[:, 0] # nonzero label index for all batch. coded class number && [2, 11]
            
            img = PIL.Image.open(os.path.join(args.voc12_root, 'JPEGImages', name_str + '.jpg'))

            # save aff image
            aff_ = np.load(args.aff_out_dir + '/' + name_str + '.npy', allow_pickle=True).item()['high_res'] # HW dim, 0 is bg, class num is already inside the image
            aff_img = np.zeros((np.unique(aff_).shape[0] - 1, aff_.shape[0], aff_.shape[1])) # 0, class numbers ... 
            
            for idx in range(1, aff_img.shape[0] + 1): 
                aff_img[idx-1][aff_ == np.unique(aff_)[idx]] = 1 # delete background
            
            aff_img_pil = []
            for channel_idx in range(aff_img.shape[0]): # cam img for each class + coloring
                aff_img_pil.append(PIL.Image.fromarray(np.uint8(cm.jet(aff_img[channel_idx, ...]) * 255)))
            for channel_idx in range(aff_img.shape[0]): # superpose on image
                plt.imshow(img, alpha = 0.5)
                plt.imshow(aff_img_pil[channel_idx], alpha = 0.4)
                plt.savefig(args.aff_out_dir + "_on_img" + '/cam_%s_%s.png' % (name_str, CAT_LIST[np.unique(aff_)[channel_idx + 1] - 1])) # CAT_LIST[valid_cat[channel_idx]]
                plt.clf()

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()