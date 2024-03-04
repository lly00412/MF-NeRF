#!/bin/bash

scenes=(horns)
losses=l2
export ROOT_DIR=/mnt/Data2/liyan/MF-NeRF/results/colmap/nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/
export CUDA_VISIBLE_DEVICES=0
for SCENE in ${scenes[@]}
do
  echo ${SCENES}
  ### mfnerf T20 128ch
  python cams_test.py \
    --val_dir ${ROOT_DIR} \
    --scene ${SCENE} \
    --random_seed 344 \
    --N_points 10 \
    --log_dir ${ROOT_DIR}/${SCENE}
  done