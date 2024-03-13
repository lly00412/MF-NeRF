#!/bin/bash

export ROOT_DIR=/mnt/Data2/nerf_datasets/nerf_llff_data/
#export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/res0.25/fewshot10/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/fewshot15_v2/
export CUDA_VISIBLE_DEVICES=1
export PREFIX=nerf_llff/Hash/res0.25/fewshot15_sr0.1

export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/res0.25/fewshot15_sr0.1

scenes=(room horns)

steps=(1 2 3 4)

for SCENES in ${scenes[@]}
do
  echo ${SCENES}
for STEP in ${steps[@]}
do

#################### entropy

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name ${PREFIX}/${SCENES}/reweighted/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --random_bg \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --ray_sampling_strategy weighted_images \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --vs_by entropy \
#    --vs_sample_rate 0.1

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --downsample 0.25 \
    --exp_name ${PREFIX}/${SCENES}/reweighted/entropy/vs${STEP}/eval/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --vs_seed 349 \
    --val_only \
    --random_bg \
    --ckpt_path ${BASE_DIR}/${SCENES}/reweighted/entropy/vs${STEP}/epoch=19.ckpt \
    --ray_sampling_strategy weighted_images \
    --pre_train_epoch 20 \
    --vs_by entropy \
    --vs_sample_rate 0.1


################# vs-nerf

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name ${PREFIX}/${SCENES}/reweighted/theta_3/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --random_bg \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --ray_sampling_strategy weighted_images \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --vs_by warp --theta 3 \
#    --vs_sample_rate 0.1

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --downsample 0.25 \
    --exp_name ${PREFIX}/${SCENES}/reweighted/theta_3/warp/vs${STEP}/eval/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --vs_seed 349 \
    --val_only \
    --random_bg \
    --ckpt_path ${BASE_DIR}/${SCENES}/reweighted/theta_3/warp/vs${STEP}/epoch=19.ckpt \
    --ray_sampling_strategy weighted_images \
    --pre_train_epoch 20 \
    --vs_by warp --theta 3 \
    --vs_sample_rate 0.1

################## random

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --random_bg \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --vs_by random \
#    --vs_sample_rate 0.1

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --downsample 0.25 \
    --exp_name ${PREFIX}/${SCENES}/random/vs${STEP}/eval/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --val_only \
    --vs_seed 349 \
    --random_bg \
    --ckpt_path ${BASE_DIR}/${SCENES}/random/vs${STEP}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --vs_by random \
    --vs_sample_rate 0.1


################## mcd_d

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name ${PREFIX}/${SCENES}/reweighted/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --random_bg \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --ray_sampling_strategy weighted_images \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --vs_by mcd_d --n_passes 30 --p 0.2 \
#    --vs_sample_rate 0.1

    python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --downsample 0.25 \
    --exp_name ${PREFIX}/${SCENES}/reweighted/mcd_d/vs${STEP}/eval/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --vs_seed 349 \
    --val_only \
    --random_bg \
    --ckpt_path ${BASE_DIR}/${SCENES}/reweighted/mcd_d/vs${STEP}/epoch=19.ckpt \
    --ray_sampling_strategy weighted_images \
    --pre_train_epoch 20 \
    --vs_by mcd_d --n_passes 30 --p 0.2 \
   --vs_sample_rate 0.1

################## mcd_r

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name ${PREFIX}/${SCENES}/reweighted/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --random_bg \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --ray_sampling_strategy weighted_images \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --vs_by mcd_r --n_passes 30 --p 0.2 \
#    --vs_sample_rate 0.1

    python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --downsample 0.25 \
    --exp_name ${PREFIX}/${SCENES}/reweighted/mcd_r/vs${STEP}/eval/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --vs_seed 349 \
    --val_only \
    --random_bg \
    --ckpt_path ${BASE_DIR}/${SCENES}/reweighted/mcd_r/vs${STEP}/epoch=19.ckpt \
    --ray_sampling_strategy weighted_images \
    --pre_train_epoch 20 \
    --vs_by mcd_r --n_passes 30 --p 0.2 \
   --vs_sample_rate 0.1

done
done
