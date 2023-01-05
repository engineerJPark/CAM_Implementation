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
from misc import torchutils
from misc.imutils import crf_inference_label, crf_inference_softmax

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

def _work(process_id, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        for iter, pack in enumerate(data_loader):
            # load cam npy
            name_str = pack['name'][0]
            label = pack['label'][0] # one hot encoded
            valid_cat = torch.nonzero(label)[:, 0] # nonzero label index for all batch
            
            img = PIL.Image.open(os.path.join(args.voc12_root, 'JPEGImages', name_str + '.jpg'))
            cam_img = np.load(args.origin_cam_dir + '/' + name_str + '.npy', allow_pickle=True).item()['high_res'] # not args.cam_out_dir
            keys = np.load(args.origin_cam_dir + '/' + name_str + '.npy', allow_pickle=True).item()['keys'] # not args.cam_out_dir
            
            # print(cam_img.shape)
            # print(np.argmax(cam_img, axis=0))
            # print(np.argmax(cam_img, axis=0).shape)
            
            print(keys)
            print(keys.shape)
            
            # # do densecrf to CAM 
            # cam_img = np.argmax(cam_img, axis=0)
            # cam_img_crf = crf_inference_label(np.asarray(img), cam_img, t=10, n_labels=keys.shape[0])
            
            # # save cams
            # np.save(os.path.join(args.cam_out_dir, name_str + '.npy'),
            #         {"keys": valid_cat, "high_res": cam_img_crf})

            # if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
            #     print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')
            
            
def run(args):
    
    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()