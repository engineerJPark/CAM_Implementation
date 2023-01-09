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
    parser.add_argument("--train_list", default="voc12/train.txt", type=str)
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
    parser.add_argument("--conf_fg_thres", default=0.30, type=float) # 0.30
    parser.add_argument("--conf_bg_thres", default=0.05, type=float) # 0.05

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="savefile/pretrained/res50_cam.pth", type=str)
    parser.add_argument("--cam_out_dir", default="savefile/result/cam", type=str) 
    parser.add_argument("--crf_out_dir", default="savefile/result/cam_crf", type=str) 
    parser.add_argument("--aff_out_dir", default="savefile/result/cam_aff", type=str) 
    parser.add_argument("--irn_weights_name", default="savefile/pretrained/res50_irn.pth", type=str) # affinity 
    parser.add_argument("--ir_label_out_dir", default="savefile/result/ir_label", type=str) # affinity 

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_aff", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)
    
    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25) # 0.25

    # Step
    parser.add_argument("--train_cam_pass", default=True)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_cam_pass", default=True)
    parser.add_argument("--draw_cam_pass", default=True)
    parser.add_argument("--make_cam_crf_pass", default=True)
    parser.add_argument("--cam_to_ir_label_pass", default=True)
    parser.add_argument("--train_cam_aff_pass", default=True) 
    parser.add_argument("--make_cam_aff_pass", default=True)
    

    args = parser.parse_args()

    os.makedirs("savefile", exist_ok=True)
    os.makedirs("savefile/result", exist_ok=True)
    os.makedirs("savefile/pretrained", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam
        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)
        
    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label
        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)
        
    if args.train_cam_aff_pass is True:
        import step.train_cam_aff
        timer = pyutils.Timer('step.train_cam_aff:')
        step.train_cam_aff.run(args)

    if args.make_cam_pass is True:
        import step.make_cam
        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)
        
    if args.make_cam_crf_pass is True:
        import step.make_cam_crf
        timer = pyutils.Timer('step.make_cam_crf:')
        step.make_cam_crf.run(args)
        
    if args.make_cam_aff_pass is True:
        import step.make_cam_aff
        timer = pyutils.Timer('step.make_cam_aff:')
        step.make_cam_aff.run(args)
        
    if args.eval_cam_pass is True:
        import step.eval_cam
        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

    if args.draw_cam_pass is True:
        import step.draw_cam
        timer = pyutils.Timer('step.draw_cam:')
        step.draw_cam.run(args)
        