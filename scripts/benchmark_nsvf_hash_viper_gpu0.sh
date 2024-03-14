#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/nerf_datasets/Synthetic_NSVF/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/fewshot10/
export CUDA_VISIBLE_DEVICES=0
export PREFIX=Synthetic_NSVF/Hash/fewshot15_sparse
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/${PREFIX}

#scenes=(Bike Lifestyle)
scenes=(Bike)

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


python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/random/eval \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --ckpt_path ${CKPT_DIR}/${SCENES}/random/vs4/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --val_only \
    --start 10 \
    --vs_by random \
    --test_img 35 \
    --vs_sample_rate 0.1

###################### vs-nerf
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
#    --start 10 --N_vs 4 --view_step 2 --epoch_step 20 \
#    --n_centers 10 \
#    --vs_by warp --theta 3 \
#    --vs_sample_rate 1.0

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/reweighted/theta_3/warp/eval \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --ckpt_path ${CKPT_DIR}/${SCENES}/reweighted/theta_3/warp/vs4/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 \
    --vs_by warp \
    --val_only \
    --test_img 35 \
    --vs_sample_rate 0.1


############################### mcd_d

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
#    --start 10 --N_vs 4 --view_step 2 --epoch_step 20 \
#    --n_centers 10 \
#    --vs_by mcd_d --n_passes 30 --p 0.2 \
#    --vs_sample_rate 1.0

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/reweighted/mcd_d/eval \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --ckpt_path ${CKPT_DIR}/${SCENES}/reweighted/mcd_d/vs4/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 \
    --val_only \
    --vs_by mcd_d \
    --test_img 35 \
    --vs_sample_rate 0.1

################################ mcd_r
#
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
#    --start 10 --N_vs 4 --view_step 2 --epoch_step 20 \
#    --n_centers 10 \
#    --vs_by mcd_r --n_passes 30 --p 0.2 \
#    --vs_sample_rate 1.0

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/reweighted/mcd_r/eval \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --ckpt_path ${CKPT_DIR}/${SCENES}/reweighted/mcd_r/vs4/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 \
    --vs_by mcd_r \
    --val_only \
    --test_img 35 \
    --vs_sample_rate 0.1

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/reweighted/entropy/eval \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --rgb_channels 128 --rgb_layers 2 \
    --grid Hash \
    --ckpt_path ${CKPT_DIR}/${SCENES}/reweighted/entropy/vs4/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 \
    --vs_by entropy \
    --val_only \
    --test_img 35 \
    --vs_sample_rate 0.1

done
