#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/nerf_datasets/
export CUDA_VISIBLE_DEVICES=0


######### lego
#
#export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NeRF/Hash/fewshot10/
#export PREFIX=Synthetic_NeRF/Hash/fewshot10
#
##scenes=(Drums)
#scenes=(Chair Ficus Hotdog)
#methods=(warp mcd_d mcd_r entropy)
#
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#
#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/Synthetic_NeRF/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/auc6/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --vs_seed 849607 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --eval_u --u_by warp mcd_d mcd_r entropy --plot_roc \
#    --theta 3 \
#    --vs_sample_rate 1.0 \
#    --val_only

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/Synthetic_NeRF/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/auc_sparse3/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/entropy/vs4/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --val_only \
#    --eval_u --u_by warp mcd_d mcd_r entropy --plot_roc \
#    --theta 3 \
#    --vs_sample_rate 0.2

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/Synthetic_NeRF/${SCENES} \
#    --dataset_name nsvf \
#    --exp_name ${PREFIX}/${SCENES}/auc_sparse5/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --vs_seed 66985 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --val_only \
#    --eval_u --u_by warp mcd_d mcd_r entropy --plot_roc \
#    --theta 3 \
#    --vs_sample_rate 0.1

#done

########## dense llff entropy
#
#export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/NGP/fewshot15/
#export PREFIX=nerf_llff/NGP/fewshot15
#
##scenes=(room horns trex fortress)
#scenes=(horns trex fortress)
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
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --vs_seed 349457 \
#    --random_bg \
#    --pre_train_epoch 20 \
#    --u_by warp mcd_r mcd_d entropy --plot_roc \
#    --theta 3 --n_passes 10 --p 0.2 \
#    --start 15 \
#    --vs_sample_rate 1.0 \
#    --eval_u \
##    --val_only \
##    --ckpt_path ${BASE_DIR}/${SCENES}/auc/epoch=19.ckpt \
#
#done

# --eval_u


######### dense llff entropy

#export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/m360/NGP/fewshot15/
#export PREFIX=m360/NGP/fewshot15

#scenes=(room horns trex fortress)
#scenes=(garden kitchen bicycle flowers)
#
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}

#################### entropy

#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/m360/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --vs_seed 349457 \
#    --random_bg \
#    --pre_train_epoch 20 \
#    --u_by warp mcd_r mcd_d entropy --plot_roc \
#    --theta 3 --n_passes 10 --p 0.2 \
#    --start 15 \
#    --vs_sample_rate 1.0 \
#    --eval_u \
#    --val_only \
#    --ckpt_path ${BASE_DIR}/${SCENES}/auc/epoch=19.ckpt \

#done

export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/LF/NGP/fewshot15/
export PREFIX=LF/NGP/fewshot15

#scenes=(room horns trex fortress)
scenes=(basket)

for SCENES in ${scenes[@]}
do
echo ${SCENES}

################### entropy

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/LF/${SCENES} \
    --dataset_name colmap \
    --downsample 1.0 \
    --exp_name ${PREFIX}/${SCENES}/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --vs_seed 349457 \
    --random_bg \
    --pre_train_epoch 20 \
    --u_by warp mcd_r mcd_d entropy \
    --n_passes 5 --p 0.2 \
    --start 15 \
    --vs_sample_rate 1.0 \
    --eval_u \
    --val_only \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \

done


export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/NGP/fewshot15/
export PREFIX=nerf_llff/NGP/fewshot15

scenes=(horns)

for SCENES in ${scenes[@]}
do
echo ${SCENES}

#################### entropy

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/nerf_llff_data/${SCENES} \
    --dataset_name colmap \
    --downsample 1.0 \
    --exp_name ${PREFIX}/${SCENES}/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --vs_seed 349457 \
    --random_bg \
    --pre_train_epoch 20 \
    --u_by warp mcd_r mcd_d entropy \
    --n_passes 5 --p 0.2 \
    --start 15 \
    --vs_sample_rate 1.0 \
    --eval_u \
    --val_only \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \

done



#export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/nsvf/Synthetic_NeRF/NGP/v15/
#export PREFIX=Synthetic_NeRF/NGP/v15/
#
#scenes=(Drums Chair Lego Ship)
#
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#
##################### entropy
#
#python train_nsvf.py \
#    --root_dir ${ROOT_DIR}/Synthetic_NeRF/${SCENES} \
#    --dataset_name nsvf \
#    --downsample 1.0 \
#    --exp_name ${PREFIX}/${SCENES}/ \
#    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --vs_seed 349457 \
#    --random_bg \
#    --pre_train_epoch 20 \
#    --u_by warp mcd_r mcd_d entropy --plot_roc \
#    --n_passes 10 --p 0.2 \
#    --start 15 \
#    --vs_sample_rate 1.0 \
#    --eval_u \
#    --val_only \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#
#done