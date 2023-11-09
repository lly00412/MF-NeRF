#!/bin/bash

export ROOT_DIR=/mnt/Data2/datasets/nerf_llff_data/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/half_res/fewshot5/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/half_res/fewshot20/
export CUDA_VISIBLE_DEVICES=0
#scenes=(fortress horns room trex)
scenes=(room)

 baseline
for SCENES in ${scenes[@]}
do
echo ${SCENES}
python train.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --downsample 0.5 \
    --exp_name nerf_llff/Hash/half_res/fewshot5/${SCENES}/ \
    --num_epochs 10 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --vs_seed 349 \
    --start 5
done

##### without vs
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
#    --weight_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --start 10 --N_vs 4 --view_step 5 --epoch_step 5 \
#    --vs_sample_rate 1.0 \
#    --no_save_vs
#done

## random vs
#for SCENES in ${scenes[@]}
#do
#echo ${SCENES}
#python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --dataset_name colmap \
#    --downsample 0.5 \
#    --exp_name nerf_llff/Hash/half_res/fewshot30_v2/${SCENES}/ \
#    --num_epochs 50 --batch_size 4096 --scale 16.0 --lr 2e-4 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#    --rgb_channels 64 --rgb_layers 2 \
#    --view_select --vs_seed 349 \
#    --start 10 --N_vs 4 --view_step 5 --epoch_step 10 \
#    --weight_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
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
    --exp_name nerf_llff/Hash/half_res/fewshot20/${SCENES}/ \
    --num_epochs 30 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 349 \
    --start 5 --N_vs 3 --view_step 5 --epoch_step 10 \
    --weight_path ${BASE_DIR}/${SCENES}/epoch=9.ckpt \
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
#    --weight_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --vs_by mcd_d --n_passes 30 --p 0.2 \
#    --vs_sample_rate 1.0
#done
#
##### mcd_r vs
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
#    --weight_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
#    --vs_by mcd_r --n_passes 30 --p 0.2 \
#    --vs_sample_rate 1.0
#done