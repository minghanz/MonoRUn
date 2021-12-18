#!/bin/sh

# ### run testing of MonoRUn model
# python test.py configs/kitti_multiclass.py kitti_multiclass.pth --val-set --gpu-ids 0 #--show-dir vis

# ### run evaluation on external results
# #_gt_conf
python eval.py configs/kitti_multiclass.py kitti_multiclass.pth /home/minghan.zhu/repos/local3d/preds_344_all_monoflex_visscore_only --result-dir /home/minghan.zhu/repos/local3d/monorun_eval/AP_preds_344_all_monoflex_visscore_only \
--val-set --gpu-ids 0 # --show-dir vis_slim

# python eval.py configs/kitti_multiclass.py kitti_multiclass.pth /home/minghan.zhu/repos/MonoFlex/output_2/toy_experiments/inference/kitti_train/data/newdet_local_ego \
# --val-set --gpu-ids 0 # --show-dir vis_slim

# python eval.py configs/kitti_multiclass.py kitti_multiclass.pth /home/minghan.zhu/repos/MonoFlex/output_0916_gt_heatmappts_bothscore/toy_experiments/inference/kitti_train/data/newdet_caronly_gtconf \
# --val-set --gpu-ids 0 # --show-dir vis_slim

###
### changelog to adapt to new version of mmdet:
### change mmdet.core to mmcv.runner for importing auto_fp16, force_fp32
### mmdet/models/roi_heads/bbox_heads/bbox_head.py line 350, add:
###         if isinstance(scale_factor, float):
###             bboxes /= scale_factor