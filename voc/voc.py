import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import collections
import matplotlib.pyplot as plt
from torchvision.datasets import VOCDetection
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensor
from typing import Any, Callable, Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# VOC 2012 dataset path
if __name__ == '__main__': 
    path2data = './'
    if not os.path.exists(path2data):
        os.mkdir(path2data)
else:
    path2data = './voc'
    if not os.path.exists(path2data):
        os.mkdir(path2data)
    
# VOC class names
classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

class myVOCDetection(VOCDetection):
    def __getitem__(self, index):
        self.normalization = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], inplace=False)
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot()) # xml-> dictionary

        targets = [] # bb coord
        labels = [] # bb classes

        # 바운딩 박스 정보 받아오기
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], classes.index(t['name'])

            targets.append(list(label[:4])) # bb coord
            labels.append(label[4])         # bb classes, use this information for CAM only

        if self.transforms:
            augmentations = self.transforms(image=img, bboxes=targets)
            ori_img = augmentations['image']
            img = self.normalization(ori_img)
            targets = augmentations['bboxes']

        labels = torch.unique(torch.tensor(labels, dtype=torch.int64), sorted=True)
        labels = F.one_hot(labels, num_classes=20)
        labels = torch.sum(labels, dim = 0) # reshape because of batch = 1
        return ori_img, img, labels


    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]: # xml-> dictionary
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

# train, validation dataset
train_ds = myVOCDetection(path2data, year='2012', image_set='train')
trainval_ds = myVOCDetection(path2data, year='2012', image_set='trainval')
val_ds = myVOCDetection(path2data, year='2012', image_set='val')

# transforms
IMAGE_SIZE = 600
scale = 1.0

# 이미지에 padding을 적용하여 종횡비를 유지시키면서 크기가 600x600 되도록 resize 합니다.
train_transforms = A.Compose([
                    A.Resize(600, 600, interpolation=2),
                    ToTensor()
                    ],
                    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                    )

val_transforms = A.Compose([
                    A.Resize(600, 600, interpolation=2),
                    ToTensor()
                    ],
                    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                    )

# train_transforms = A.Compose([
#                     A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
#                     A.PadIfNeeded(min_height=int(IMAGE_SIZE*scale), min_width=int(IMAGE_SIZE*scale),border_mode=cv2.BORDER_CONSTANT),
#                     ToTensor()
#                     ],
#                     bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
#                     )

# val_transforms = A.Compose([
#                     A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
#                     A.PadIfNeeded(min_height=int(IMAGE_SIZE*scale), min_width=int(IMAGE_SIZE*scale),border_mode=cv2.BORDER_CONSTANT),
#                     ToTensor()
#                     ],
#                     bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
#                     )

# apply transforms
train_ds.transforms = train_transforms
trainval_ds.transforms = train_transforms
val_ds.transforms = val_transforms

# dataloader definition
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
trainval_dl = DataLoader(trainval_ds, batch_size=1, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
