clear
python main.py \
--draw_cam_pass=False \
--draw_crf_pass=False \
--draw_aff_pass=False \
--train_cam_pass=False \
--make_cam_pass=False \
--make_crf_pass=False \
--cam_eval_thres=0.15 \
--alpha=16 \
--conf_fg_thres=0.40 \
--conf_bg_thres=0.10 \
--ins_seg_bg_thres=0.25 \
--sem_seg_bg_thres=0.25 \

# --eval_cam_pass=False \
# --eval_crf_pass=False \
# --cam_to_ir_label_pass=False \
# --train_aff_pass=False \
# --make_aff_pass=False \
# --make_irn_pass=False \
# --eval_aff_pass=False \
# --eval_irn_pass=False \
# --draw_irn_pass=False \