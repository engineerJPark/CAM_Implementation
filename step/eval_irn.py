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
        mask_list4img= []
        for i in np.unique(ins_out['class']):
            mask_list = []
            for j in range(len(ins_out['class'])):
                if i == ins_out['class'][j]:
                    ins_out['mask'][j][ins_out['mask'][j] == 255] = 0
                    mask_list.append(ins_out['mask'][j])
            mask_list4img.append(np.stack(mask_list, axis=0).sum(axis=0) * (i + 1)) ## wait ...
        pred_mask.append(np.stack(mask_list4img, axis=0))
    
    pred_mask = np.asarray(pred_mask)
    labels = np.asarray(labels)
    
    ## need to make as True False to 0,1,14 ...

    # print(np.unique(preds[0]))
    # print(preds[0].shape) ## (281, 500)
    # print(preds.ndim) ## 1
    # print(labels.ndim) ## 1
    # print(len(preds)) ## 1
    # print(len(labels)) ## 1
    # print(pred.shape) ## (1464,)
    # print(labels.shape) ## (1464,)
    # print(pred_mask.ndim) ## 1
    # print(labels.ndim) ## 1
    
    # print(pred_mask[0].shape) ## (1464,)
    # print(labels[0].shape) ## (1464,)
    
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
