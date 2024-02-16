#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/nerf_datasets/Synthetic_NeRF/
export BASE_DIR=~/mnt/Data2/liyan/MF-NeRF/ckpts/Synthetic_NeRF/Hash/fewshot10/
export CKPT_DIR=~/mnt/Data2/liyan/MF-NeRF/ckpts/Synthetic_NeRF/Hash/fewshot11/
export CUDA_VISIBLE_DEVICES=0
export PREFIX=Synthetic_NeRF/Hash/fewshot11

#export ROOT_DIR=/media/landa/lchen39/datasets/Synthetic_NeRF/
#export BASE_DIR=~/projects/MF-NeRF/ckpts/nsvf/Synthetic_NeRF/Hash/fewshot10/
#export CKPT_DIR=~/projects/MF-NeRF/ckpts/nsvf/Synthetic_NeRF/Hash/fewshot11/
#export CUDA_VISIBLE_DEVICES=0
#export PREFIX=Synthetic_NeRF/Hash/fewshot11/

scenes=(Hotdog Chair Drums Ficus)

for SCENES in ${scenes[@]}
do
echo ${SCENES}

python data_generater.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name Synthetic_NeRF/Hash/fewshot10/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --grid Hash \
    --rgb_channels 128 --rgb_layers 2 \
    --vs_seed 66985 \
    --pre_train_epoch 20 \
    --start 10 --N_more 0 \
    --train_imgs 57 27 32 63 92 19 85 40 20 69 \
    --val_only \
    --eval_u --u_by warp --theta 3 \
    --vs_sample_rate 0.2

python data_generater.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --grid Hash \
    --rgb_channels 128 --rgb_layers 2 \
    --vs_seed 66985 \
    --pre_train_epoch 20 \
    --start 10 --N_more 20 \
    --train_imgs 57 27 32 63 92 19 85 40 20 69 \
    --val_only \
    --eval_u --u_by warp --theta 3 \
    --vs_sample_rate 0.2

done