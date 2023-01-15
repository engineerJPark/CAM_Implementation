import numpy as np
import os

import chainercv
from chainercv.datasets import VOCInstanceSegmentationDataset, VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    # dataset = VOCInstanceSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    print("cam after irn\n")
    pred_mask = []
    for id in dataset.ids:
        ins_out = np.load(os.path.join(args.irn_out_dir, id + '.npy'), allow_pickle=True).item()
        img_per_mask = np.zeros_like(ins_out['mask'][0], dtype=np.int64)
        for i in np.unique(ins_out['class']):
            ith_mask = np.full_like(ins_out['mask'][0], False)
            for j in range(len(ins_out['class'])):
                if i == ins_out['class'][j]:
                    ith_mask += ins_out['mask'][j]
            img_per_mask += ith_mask * (i+1)
        pred_mask.append(np.stack(img_per_mask, axis=0))

    pred_mask = np.asarray(pred_mask)
    labels = np.asarray(labels)
    
    confusion = calc_semantic_segmentation_confusion(pred_mask, labels)[:21, :21]
    
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    print({'iou': iou, 'miou': np.nanmean(iou)})
