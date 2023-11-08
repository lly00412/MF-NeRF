#!/bin/bash

export ROOT_DIR=/mnt/Data2/datasets/nerf_llff_data/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/half_res/fewshot30/
export CUDA_VISIBLE_DEVICES=0
#scenes=(fortress horns room trex)
scenes=(room)

#### without vs
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.5 \
#    --exp_name nerf_llff/Hash/half_res/fewshot30/${SCENES}/ \
#    --num_epochs 25 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --start 10 --N_vs 4 --view_step 5 --epoch_step 5 \
#    --vs_sample_rate 1.0 \
#    --no_save_vs
#done

#### random vs
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.5 \
#    --exp_name nerf_llff/Hash/half_res/fewshot30/${SCENES}/ \
#    --num_epochs 25 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --start 10 --N_vs 4 --view_step 5 --epoch_step 5 \
#    --vs_by random \
#    --vs_sample_rate 1.0 \
#    --no_save_vs
#done

## warp vs
for SCENES in ${scenes[@]}
do
echo ${SCENES}
python train.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --downsample 0.5 \
    --exp_name nerf_llff/Hash/half_res/fewshot30/${SCENES}/ \
    --num_epochs 25 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 349 \
    --start 10 --N_vs 4 --view_step 5 --epoch_step 5 \
    --vs_by warp --theta 1 \
    --vs_sample_rate 1.0
done

#### mcd_d vs
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.5 \
#    --exp_name nerf_llff/Hash/half_res/fewshot30/${SCENES}/ \
#    --num_epochs 25 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --start 10 --N_vs 4 --view_step 5 --epoch_step 5 \
#    --vs_by mcd_d --n_passes 30 --p 0.2 \
#    --vs_sample_rate 1.0
#done

#### mcd_r vs
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.5 \
#    --exp_name nerf_llff/Hash/half_res/fewshot30/${SCENES}/ \
#    --num_epochs 25 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --start 10 --N_vs 4 --view_step 5 --epoch_step 5 \
#    --vs_by mcd_r --n_passes 30 --p 0.2 \
#    --vs_sample_rate 1.0
#done