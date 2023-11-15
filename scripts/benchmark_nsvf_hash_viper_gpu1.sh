#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/datasets/Synthetic_NSVF/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/${losses}/fewshot50/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/fewshot70/
export CUDA_VISIBLE_DEVICES=1

scenes=(Wineholder Steamtrain Toad Palace)

##### mcd_d vs
for SCENES in ${scenes[@]}
do
echo ${SCENES}
###################  mcd_d vs

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name Synthetic_NSVF/Hash/fewshot70/${SCENES}/reweighted/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --view_select --vs_seed 349 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --ray_sampling_strategy weighted_images \
    --pre_train_epoch 20 \
    --start 50 --N_vs 4 --view_step 5 --epoch_step 20 \
    --vs_by mcd_d --n_passes 30 --p 0.2 \
    --vs_sample_rate 1.0

###################  mcd_r vs

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name Synthetic_NSVF/Hash/fewshot70/${SCENES}/reweighted/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --view_select --vs_seed 349 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --ray_sampling_strategy weighted_images \
    --pre_train_epoch 20 \
    --start 50 --N_vs 4 --view_step 5 --epoch_step 20 \
    --vs_by mcd_r --n_passes 30 --p 0.2 \
    --vs_sample_rate 1.0

done