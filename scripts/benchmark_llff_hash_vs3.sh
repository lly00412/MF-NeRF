#!/bin/bash

export ROOT_DIR=/mnt/Data2/datasets/nerf_llff_data/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/res0.25/fewshot10/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/fewshot20/
export CUDA_VISIBLE_DEVICES=1
scenes=(trex leaves orchids)

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
for SCENES in ${scenes[@]}
do
echo ${SCENES}
python train_v2.py \
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
    --vs_by warp --theta 1 \
    --vs_sample_rate 1.0
done