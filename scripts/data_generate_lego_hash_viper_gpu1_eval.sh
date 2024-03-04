#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/nerf_datasets/nerf_llff_data/
export BASE_DIR=~/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/Hash/res0.25/fewshot10/
export CKPT_DIR=~/mnt/Data2/liyan/MF-NeRF/ckpts/Synthetic_NeRF/Hash/fewshot11/
export CUDA_VISIBLE_DEVICES=1
export PREFIX=nerf_llff/Hash/res0.25/

scenes=(fortress horns room trex)

SCENES=fortress
echo ${SCENES}

python data_generater.py --root_dir ${ROOT_DIR}/${SCENES} --dataset_name colmap --downsample 0.25 \
    --exp_name ${PREFIX}/fewshot10/${SCENES}/ \
    --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --rgb_channels 64 --rgb_layers 2 --random_bg \
    --vs_seed 349378 --start 10 --N_more 20 --train_imgs 21 19 18 17 34 20 28 16 27 31 \
    --vs_sample_rate 1.0 --val_only --view_select --vs_by warp --theta 3 --u_hist

python data_generater.py --root_dir ${ROOT_DIR}/${SCENES} --dataset_name colmap --downsample 0.25 \
    --exp_name ${PREFIX}/fewshot11/${SCENES}/ \
    --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --rgb_channels 64 --rgb_layers 2 --random_bg \
    --vs_seed 349378 --start 10 --N_more 20 --train_imgs 21 19 18 17 34 20 28 16 27 31 \
    --vs_sample_rate 1.0 --val_only

SCENES=horns
echo ${SCENES}

python data_generater.py --root_dir ${ROOT_DIR}/${SCENES} --dataset_name colmap --downsample 0.25 \
    --exp_name ${PREFIX}/fewshot10/${SCENES}/ \
    --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --rgb_channels 64 --rgb_layers 2 --random_bg \
    --vs_seed 349378 --start 10 --N_more 20 --train_imgs 43 53 24 51 47 20 22 29 17 13 \
    --vs_sample_rate 1.0 --val_only --view_select --vs_by warp --theta 3 --u_hist

python data_generater.py --root_dir ${ROOT_DIR}/${SCENES} --dataset_name colmap --downsample 0.25 \
    --exp_name ${PREFIX}/fewshot11/${SCENES}/ \
    --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --rgb_channels 64 --rgb_layers 2 --random_bg \
    --vs_seed 349378 --start 10 --N_more 20 --train_imgs 43 53 24 51 47 20 22 29 17 13 \
    --vs_sample_rate 1.0 --val_only

SCENES=room
echo ${SCENES}

python data_generater.py --root_dir ${ROOT_DIR}/${SCENES} --dataset_name colmap --downsample 0.25 \
    --exp_name ${PREFIX}/fewshot10/${SCENES}/ \
    --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --rgb_channels 64 --rgb_layers 2 --random_bg \
    --vs_seed 349378 --start 10 --N_more 20 --train_imgs 21 19 18 17 33 20 28 16 27 31 \
    --vs_sample_rate 1.0 --val_only --view_select --vs_by warp --theta 3 --u_hist

python data_generater.py --root_dir ${ROOT_DIR}/${SCENES} --dataset_name colmap --downsample 0.25 \
    --exp_name ${PREFIX}/fewshot11/${SCENES}/ \
    --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --rgb_channels 64 --rgb_layers 2 --random_bg \
    --vs_seed 349378 --start 10 --N_more 20 --train_imgs 21 19 18 17 33 20 28 16 27 31 \
    --vs_sample_rate 1.0 --val_only

SCENES=trex
echo ${SCENES}

python data_generater.py --root_dir ${ROOT_DIR}/${SCENES} --dataset_name colmap --downsample 0.25 \
    --exp_name ${PREFIX}/fewshot10/${SCENES}/ \
    --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --rgb_channels 64 --rgb_layers 2 --random_bg \
    --vs_seed 349378 --start 10 --N_more 20 --train_imgs 47 13 17 46 22 20 29 18 32 28 \
    --vs_sample_rate 1.0 --val_only --view_select --vs_by warp --theta 3 --u_hist

python data_generater.py --root_dir ${ROOT_DIR}/${SCENES} --dataset_name colmap --downsample 0.25 \
    --exp_name ${PREFIX}/fewshot11/${SCENES}/ \
    --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash --rgb_channels 64 --rgb_layers 2 --random_bg \
    --vs_seed 349378 --start 10 --N_more 20 --train_imgs 47 13 17 46 22 20 29 18 32 28 \
    --vs_sample_rate 1.0 --val_only