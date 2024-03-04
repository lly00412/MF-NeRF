#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/datasets/Synthetic_NSVF/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/fewshot10/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/fewshot15_sparse/
export CUDA_VISIBLE_DEVICES=1
export PREFIX=Synthetic_NSVF/Hash/fewshot15_v2


scenes=(Robot)
#scenes=(Bike Lifestyle Wineholder)

for SCENES in ${scenes[@]}
do
echo ${SCENES}

#################### entropy

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/reweighted/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --view_select --vs_seed 66985 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
    --n_centers 10 \
    --ray_sampling_strategy weighted_images \
    --vs_by entropy \
    --vs_sample_rate 1.0

################# vs-nerf

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/reweighted/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --rgb_channels 128 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --ray_sampling_strategy weighted_images \
#    --vs_by warp --theta 1 \
#    --vs_sample_rate 1.0

################## random

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --rgb_channels 128 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --vs_by random \
#    --vs_sample_rate 1.0

################## mcd_d

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/reweighted/30ps/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --rgb_channels 128 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --ray_sampling_strategy weighted_images \
#    --vs_by mcd_d --n_passes 30 --p 0.2 \
#    --vs_sample_rate 1.0
#
#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/reweighted/10ps/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --rgb_channels 128 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --ray_sampling_strategy weighted_images \
#    --vs_by mcd_d --n_passes 10 --p 0.2 \
#    --vs_sample_rate 1.0

################## mcd_r

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/reweighted/30ps/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --rgb_channels 128 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --ray_sampling_strategy weighted_images \
#    --vs_by mcd_r --n_passes 30 --p 0.2 \
#    --vs_sample_rate 1.0
#
#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/reweighted/10ps/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --rgb_channels 128 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --ray_sampling_strategy weighted_images \
#    --vs_by mcd_r --n_passes 10 --p 0.2 \
#    --vs_sample_rate 1.0

done