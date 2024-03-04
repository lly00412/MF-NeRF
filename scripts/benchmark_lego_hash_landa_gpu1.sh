#!/bin/bash

losses=l2
export ROOT_DIR=/media/landa/lchen39/datasets/Synthetic_NeRF/
export BASE_DIR=~/projects/MF-NeRF/ckpts/nsvf/Synthetic_NeRF/Hash/fewshot10/
export CKPT_DIR=~/projects/MF-NeRF/ckpts/nsvf/Synthetic_NeRF/Hash/fewshot20/
export CUDA_VISIBLE_DEVICES=1
export PREFIX=Synthetic_NeRF/Hash/fewshot20_rebuttal/

scenes=(Chair)
for SCENES in ${scenes[@]}
do
echo ${SCENES}


######baseline

#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name Synthetic_NSVF/Hash/fewshot10/${SCENES}/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --rgb_channels 128 --rgb_layers 2 \
#    --grid Hash \
#    --start 10 \
#    --vs_seed 66985 --no_save_vs

######################### random vs

###################### vs-nerf
#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --vs_by warp --theta 3 \
#    --vs_sample_rate 1.0

####### start uniform

echo Uniform

#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name Synthetic_NeRF/Hash/fewshot10/${SCENES}/Uniform/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --rgb_channels 128 --rgb_layers 2 \
#    --grid Hash \
#    --start 10 \
#    --vs_seed 66985 --no_save_vs \
#    --train_img 31 10 2 91 48 4 33 45 50 49


python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/Uniform/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --grid Hash \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --view_select --vs_seed 66985 \
    --ckpt_path ${BASE_DIR}/${SCENES}/Uniform/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
    --n_centers 10 \
    --vs_by warp --theta 3 \
    --vs_sample_rate 1.0 \
    --train_img 31 10 2 91 48 4 33 45 50 49

echo Front

#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name Synthetic_NeRF/Hash/fewshot10/${SCENES}/Front/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --rgb_channels 128 --rgb_layers 2 \
#    --grid Hash \
#    --start 10 \
#    --vs_seed 66985 --no_save_vs \
#    --train_img 1 8 14 32 36 61 67 88 92 98


python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/Front/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --grid Hash \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --view_select --vs_seed 66985 \
    --ckpt_path ${BASE_DIR}/${SCENES}/Front/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
    --n_centers 10 \
    --vs_by warp --theta 3 \
    --vs_sample_rate 1.0 \
    --train_img 1 8 14 32 36 61 67 88 92 98





######### random

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --vs_by random \
#    --no_save_vs \
#    --vs_sample_rate 0.1

############################### mcd_d

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --vs_by mcd_d --n_passes 30 --p 0.2 \
#    --vs_sample_rate 0.1
#
################################ mcd_r
#
#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --grid Hash \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
#    --n_centers 10 \
#    --vs_by mcd_r --n_passes 30 --p 0.2 \
#    --vs_sample_rate 0.1
#
done