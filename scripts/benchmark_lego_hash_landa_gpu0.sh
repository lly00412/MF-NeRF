#!/bin/bash

losses=l2
export ROOT_DIR=/media/landa/lchen39/datasets/Synthetic_NeRF/
export BASE_DIR=~/projects/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/fewshot10/
export CKPT_DIR=~/projects/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/fewshot20/
export CUDA_VISIBLE_DEVICES=0
export PREFIX=Synthetic_NSVF/Hash/fewshot20_v1

scenes=(Hotdog)
for SCENES in ${scenes[@]}
do
echo ${SCENES}
########## random
python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 66985 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 --N_vs 5 --view_step 1 --epoch_step 20 \
    --vs_by random \
    --no_save_vs \
    --vs_sample_rate 1.0

########### warp
python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 66985 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 --N_vs 5 --view_step 1 --epoch_step 20 \
    --vs_by warp --theta 3 \
    --vs_sample_rate 1.0
done

scenes=(Chair Drums Ficus Hotdog)

for SCENES in ${scenes[@]}
do
echo ${SCENES}

######################### random vs
#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 5 --view_step 1 --epoch_step 20 \
#    --vs_by random \
#    --no_save_vs \
#    --vs_sample_rate 1.0
#
####################### warp vs
#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 5 --view_step 1 --epoch_step 20 \
#    --vs_by warp --theta 3 \
#    --vs_sample_rate 1.0

############################### mcd_d

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 66985 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 --N_vs 5 --view_step 1 --epoch_step 20 \
    --vs_by mcd_d --n_passes 30 --p 0.2 \
    --vs_sample_rate 1.0

############################### mcd_r

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 66985 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 --N_vs 5 --view_step 1 --epoch_step 20 \
    --vs_by mcd_r --n_passes 30 --p 0.2 \
    --vs_sample_rate 1.0

done
