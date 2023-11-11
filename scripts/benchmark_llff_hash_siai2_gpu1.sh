#!/bin/bash

export ROOT_DIR=/data/datasets/nerf_llff_data/
export BASE_DIR=~/Desktop/projects/MF-NeRF/ckpts/nerf_llff/Hash/res0.25/fewshot10/
export CKPT_DIR=~/Desktop/projects/MF-NeRF/ckpts/nerf_llff/Hash/fewshot20/
export CUDA_VISIBLE_DEVICES=0
scenes=(fortress horns leaves)
#scenes=(room)

######baseline
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name nerf_llff/Hash/res0.25/fewshot10/${SCENES}/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --random_bg \
#    --vs_seed 349 \
#    --start 10
#done
#
##### without vs
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --exp_name nerf_llff/Hash/fewshot20/${SCENES}/ \
#    --num_epochs 40 --batch_size 4096 --scale 16.0 --lr 2e-3 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --random_bg \
#    --start 5 --N_vs 3 --view_step 5 --epoch_step 10 \
#    --weight_path ${BASE_DIR}/${SCENES}/epoch=9.ckpt \
#    --vs_sample_rate 1.0 \
#    --no_save_vs
#done
#
######################### random vs
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#python train_v2.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name nerf_llff/Hash/res0.25/fewshot20_v2/${SCENES}/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --random_bg \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 2 --epoch_step 20 \
#    --vs_by random \
#    --vs_sample_rate 1.0
#done
#
############ warp vs
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#python train_v2.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.25 \
#    --exp_name nerf_llff/Hash/res0.25/fewshot20_v2/${SCENES}/ \
#    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --random_bg \
#    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --pre_train_epoch 20 \
#    --start 10 --N_vs 4 --view_step 2 --epoch_step 20 \
#    --vs_by warp --theta 1 \
#    --vs_sample_rate 1.0
#done
#
### mcd_d vs
for SCENES in ${scenes[@]}
do
echo ${SCENES}
python python train_v2.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --downsample 0.25 \
    --exp_name nerf_llff/Hash/res0.25/fewshot20_v2/${SCENES}/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 349 \
    --random_bg \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 --N_vs 4 --view_step 2 --epoch_step 20 \
    --vs_by mcd_d --n_passes 30 --p 0.2 \
    --vs_sample_rate 1.0
done
#
##### mcd_r vs
for SCENES in ${scenes[@]}
do
echo ${SCENES}
python python train_v2.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --downsample 0.25 \
    --exp_name nerf_llff/Hash/res0.25/fewshot20_v2/${SCENES}/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 349 \
    --random_bg \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --start 10 --N_vs 4 --view_step 2 --epoch_step 20 \
    --vs_by mcd_r --n_passes 30 --p 0.2 \
    --vs_sample_rate 1.0
done