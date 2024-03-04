#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/nerf_datasets/
export CUDA_VISIBLE_DEVICES=0


######### dense lego entropy

export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NeRF/Hash/fewshot10_v2/
export PREFIX=Synthetic_NeRF/Hash/fewshot15

scenes=(Hotdog Chair Drums Ficus)

for SCENES in ${scenes[@]}
do
echo ${SCENES}

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/Synthetic_NeRF/${SCENES} \
    --dataset_name nsvf \
    --exp_name Synthetic_NeRF/Hash/fewshot10_v2/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --vs_seed 66985 \
    --start 10 \
    --n_centers 10 \
    --vs_by entropy \
    --vs_sample_rate 1.0


python train_nsvf.py \
    --root_dir ${ROOT_DIR}/Synthetic_NeRF/${SCENES} \
    --dataset_name nsvf \
    --exp_name ${PREFIX}/${SCENES}/ \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 66985 \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --pre_train_epoch 20 \
    --ray_sampling_strategy weighted_images \
    --start 10 --N_vs 4 --view_step 1 --epoch_step 20 \
    --n_centers 10 \
    --vs_by entropy \
    --vs_sample_rate 1.0

done

######### dense llff entropy

export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/res0.25/fewshot10/
export PREFIX=nerf_llff/Hash/fewshot15
scenes=(room horns trex fortress)

for SCENES in ${scenes[@]}
do
echo ${SCENES}

#################### entropy

python train_nsvf.py \
    --root_dir ${ROOT_DIR}/nerf_llff_data/${SCENES} \
    --dataset_name colmap \
    --downsample 0.25 \
    --exp_name ${PREFIX}/${SCENES}/reweighted/ \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 \
    --view_select --vs_seed 349 \
    --random_bg \
    --ckpt_path ${BASE_DIR}/${SCENES}/epoch=19.ckpt \
    --ray_sampling_strategy weighted_images \
    --pre_train_epoch 20 \
    --start 10 --N_vs 5 --view_step 1 --epoch_step 20 \
    --vs_by entropy \
    --vs_sample_rate 1.0

done