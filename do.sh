clear
CUDA_VISIBLE_DEVICES=0 python main.py \
--irn_batch_size=16 \
--irn_learning_rate=0.15 \
--train_cam_pass=False \
--make_cam_pass=False \
--eval_cam_pass=False \
--make_crf_pass=False \
--eval_crf_pass=False \
--cam_to_ir_label_pass=False \
# --train_aff_pass=False \
# --make_aff_pass=False \
# --eval_aff_pass=False \ 

# clear
# python main.py \
# --irn_batch_size=32 \
# --irn_learning_rate=0.1 \
# --train_cam_pass=False \
# --make_cam_pass=False \
# --eval_cam_pass=False \
# --make_crf_pass=False \
# --eval_crf_pass=False \
# --cam_to_ir_label_pass=False \