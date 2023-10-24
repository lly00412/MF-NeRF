#!/bin/bash

scenes=(fern)
losses=l2
export ROOT_DIR=/mnt/Data2/datasets/nerf_llff_data/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/
export CUDA_VISIBLE_DEVICES=0
#for SCENES in ${scenes[@]}
#do
#  echo ${SCENES}
#  ### mfnerf T20 128ch
#  python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/norm_cams2/${SCENES}/ \
#    --num_epochs 10 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --N_tables 8 \
#    --rgb_channels 128 --rgb_layers 2 \
#    --loss ${losses}
#  done

for SCENES in ${scenes[@]}
do
  echo ${SCENES}
  ### mfnerf T20 128ch
  export CUDA_LAUNCH_BLOCKING=1
  export TORCH_USE_CUDA_DSA=1
  python train.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/norm_cams2/${SCENES}/ \
    --num_epochs 20 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --N_tables 8 \
    --rgb_channels 128 --rgb_layers 2 \
    --loss ${losses} \
    --val_only --ckpt ${CKPT_DIR}/norm_cams2/${SCENES}/epoch=9.ckpt \
    --mcdropout --n_passes 30 --p 0.5 \
    --warp --ref_cam 0 \
    --save_output \
    --plot_roc
  done


# --mcdropout --n_passes 30 --p 0.2 \

#--val_only --ckpt ${CKPT_DIR}/norm_cams/${SCENES}/epoch=19.ckpt \
#    --mcdropout --n_passes 30 --p 0.5 \
#    --warp --ref_cam 0 \
#    --save_output \
#    --plot_roc