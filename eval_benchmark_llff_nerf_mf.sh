#!/bin/bash

export ROOT_DIR=/mnt/Data2/liyan/MF-NeRF/results
export EXP_NAME=colmap/nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/fewshot/


python eval.py --val_dir ${ROOT_DIR}/${EXP_NAME} \
      --scenes fern/mcd100 flower/mcd100 fortress/mcd100 horns/mcd100 leaves/mcd100 orchids/mcd100 room/mcd100 trex/mcd100 \
      --opt err \
      --est mcd psnr ssim flip \
      --plot_roc
#      --plot_metric