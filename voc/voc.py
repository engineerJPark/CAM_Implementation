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

# VOC 2012 dataset을 저장할 위치
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
            img = augmentations['image']
            targets = augmentations['bboxes']

        return img, targets, labels

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

# train, validation dataset을 생성합니다.
train_ds = myVOCDetection(path2data, year='2012', image_set='train')
trainval_ds = myVOCDetection(path2data, year='2012', image_set='trainval')
val_ds = myVOCDetection(path2data, year='2012', image_set='val')


#######################################################################
#######################################################################
# 샘플 이미지 확인
img, target, label = train_ds[3]
colors = np.random.randint(0, 255, size=(80,3), dtype='uint8') # 바운딩 박스 색상

# 시각화 함수
def show(img, targets, labels, classes=classes):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    targets = np.array(targets)
    W, H = img.size

    for tg,label in zip(targets,labels):
        id_ = int(label) # class
        bbox = tg[:4]    # [x1, y1, x2, y2]

        color = [int(c) for c in colors[id_]]
        name = classes[id_]

        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=tuple(color), width=3)
        draw.text((bbox[0], bbox[1]), name, fill=(255,255,255,0))
    plt.imshow(np.array(img))
    plt.savefig('testimg01.png')

plt.figure(figsize=(10,10))
show(img, target, label)

#######################################################################
#######################################################################
# transforms 정의
IMAGE_SIZE = 600
scale = 1.0

# 이미지에 padding을 적용하여 종횡비를 유지시키면서 크기가 600x600 되도록 resize 합니다.
train_transforms = A.Compose([
                    A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
                    A.PadIfNeeded(min_height=int(IMAGE_SIZE*scale), min_width=int(IMAGE_SIZE*scale),border_mode=cv2.BORDER_CONSTANT),
                    ToTensor()
                    ],
                    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                    )

val_transforms = A.Compose([
                    A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
                    A.PadIfNeeded(min_height=int(IMAGE_SIZE*scale), min_width=int(IMAGE_SIZE*scale),border_mode=cv2.BORDER_CONSTANT),
                    ToTensor()
                    ],
                    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                    )

# transforms 적용하기
train_ds.transforms = train_transforms
val_ds.transforms = val_transforms

#######################################################################
#######################################################################
# dataloader definition
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
trainval_dl = DataLoader(trainval_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

# print(val_ds)
# print(val_dl)