#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/datasets/Synthetic_NSVF/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/${losses}/fewshot50/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/fewshot70/
export CUDA_VISIBLE_DEVICES=0

#scenes=(Wineholder Steamtrain Toad Palace)
scenes=(Toad Palace)


######baseline
for SCENES in ${scenes[@]}
do
echo ${SCENES}
python train.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name Synthetic_NSVF/Hash/l2/fewshot50/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --start 50 \
    --vs_seed 349 --no_save_vs
done

scenes=(Wineholder Steamtrain Toad Palace)

######################### random vs
for SCENES in ${scenes[@]}
do
echo ${SCENES}
python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name Synthetic_NSVF/Hash/fewshot70/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --view_select --vs_seed 349 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 50 --N_vs 4 --view_step 5 --epoch_step 20 \
    --vs_by random \
    --no_save_vs \
    --vs_sample_rate 1.0
done

#scenes=(Steamtrain Toad Palace)
#
thetas=(5)
#
############ warp vs
for SCENES in ${scenes[@]}
do
echo ${SCENES}
for T in ${thetas[@]}
do
python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name Synthetic_NSVF/Hash/fewshot70/${SCENES}/reweighted/theta_${T}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --view_select --vs_seed 349 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --ray_sampling_strategy weighted_images \
    --pre_train_epoch 20 \
    --start 50 --N_vs 4 --view_step 5 --epoch_step 20 \
    --vs_by warp --theta ${T} \
    --vs_sample_rate 1.0
done
done
