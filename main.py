import argparse
import os

from misc import pyutils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--voc12_root", default='dataset/VOCdevkit/VOC2012', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="savefile/pretrained/res50_cam.pth", type=str)
    parser.add_argument("--cam_out_dir", default="savefile/result/cam", type=str)
    parser.add_argument("--cam_on_img_dir", default="savefile/result/cam_on_img", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=True)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_cam_pass", default=True)
    parser.add_argument("--draw_cam_pass", default=True)
    # parser.add_argument("--densecrf_pass", default=True)
    # parser.add_argument("--affinity_pass", default=True)

    args = parser.parse_args()

    os.makedirs("savefile", exist_ok=True)
    os.makedirs("savefile/result", exist_ok=True)
    os.makedirs("savefile/pretrained", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.cam_on_img_dir, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam
        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)

    if args.make_cam_pass is True:
        import step.make_cam
        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam
        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

    if args.draw_cam_pass is True:
        import step.draw_cam
        timer = pyutils.Timer('step.draw_cam:')
        step.draw_cam.run(args)
        
    # if args.densecrf_pass is True:
    #     import step.densecrf_pass
    #     timer = pyutils.Timer('step.densecrf_pass:')
    #     step.densecrf_pass.run(args)
        
    # if args.affinity_pass is True:
    #     import step.affinity_pass
    #     timer = pyutils.Timer('step.v:')
    #     step.affinity_pass.run(args)

# python main.py --train_cam_pass=False --make_cam_pass=False --eval_cam_pass=False