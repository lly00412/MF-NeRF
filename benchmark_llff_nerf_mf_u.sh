#!/bin/bash

scenes=(fern)
losses=l2
export ROOT_DIR=/mnt/Data2/datasets/nerf_llff_data/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/fewshot
export CUDA_VISIBLE_DEVICES=0
for SCENES in ${scenes[@]}
do
  echo ${SCENES}
  ### mfnerf T20 128ch
  export CUDA_LAUNCH_BLOCKING=1
  export TORCH_USE_CUDA_DSA=1
  python train.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/fewshot/${SCENES}/ \
    --num_epochs 10 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
    --rgb_channels 128 --rgb_layers 2 \
    --loss ${losses} \
    --fewshot 10 \
    --save_output \
    --val_only --ckpt_path ${CKPT_DIR}/${SCENES}/epoch=9.ckpt \
    --warp
  done