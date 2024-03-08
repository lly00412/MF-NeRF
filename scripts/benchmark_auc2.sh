#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/nerf_datasets/
export CUDA_VISIBLE_DEVICES=1


######### lego
#
######### lego
#
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NeRF/Hash/fewshot15/
export PREFIX=Synthetic_NeRF/Hash/fewshot15

scenes=(Hotdog Chair Drums Ficus)

for SCENES in ${scenes[@]}
do
echo ${SCENES}

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/Synthetic_NeRF/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/auc3/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --rgb_channels 128 --rgb_layers 2 \
#    --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/entropy/vs4/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --val_only \
#    --eval_u --u_by warp mcd_d mcd_r entropy --plot_roc \
#    --theta 3 \
#    --vs_sample_rate 1.0

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/Synthetic_NeRF/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/auc_sparse3/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --vs_seed 66985 \
    --ckpt_path ${BASE_DIR}/${SCENES}/entropy/vs4/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --val_only \
    --eval_u --u_by warp mcd_d mcd_r entropy --plot_roc \
    --theta 3 \
    --no_save_test \
    --vs_sample_rate 0.2


done


######### dense llff entropy

#export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/res0.25/fewshot10/
#export PREFIX=nerf_llff/Hash/fewshot15
#
#scenes=(fortress)
#
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#
##################### entropy
#
#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/nerf_llff_data/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name ${PREFIX}/${SCENES}/auc_sr2/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --val_only \
#    --vs_seed 349486 \
#    --random_bg \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --eval_u --u_by mcd_d mcd_r --plot_roc \
#    --theta 3 --n_passes 30 --p 0.2 \
#    --start 10 \
#    --vs_sample_rate 0.2
#
#done