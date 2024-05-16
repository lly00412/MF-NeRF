#!/bin/bash

export ROOT_DIR=/mnt/Data2/nerf_datasets/tanks_and_temples
export CUDA_VISIBLE_DEVICES=1
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nerfpp/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/splitnerfpp/
export PREFIX=splitnerfpp/tat_intermediate_M60_split/fewshot36/

#python train_nsvf.py \
#    --root_dir $ROOT_DIR/tat_intermediate_M60_split --dataset_name splitnerfpp \
#    --exp_name tat_intermediate_M60 \
#    --num_epochs 20 --scale 4.0 \
#    --grid Hash \

## baseline

#python train.py \
#    --root_dir $ROOT_DIR/tat_intermediate_M60_split --dataset_name splitnerfpp \
#    --exp_name tat_intermediate_M60_split --eval_lpips \
#    --num_epochs 20 --scale 4.0 \
#    --grid Hash \
#    --start 20 \
#    --vs_seed 37661 --no_save_vs
SCENES=tat_intermediate_M60

######### entropy #######################
#python train_nsvf.py \
#    --root_dir $ROOT_DIR/${SCENES} --dataset_name splitnerfpp \
#    --exp_name ${SCENES}/fewshot42/ --eval_lpips \
#    --num_epochs 20 --scale 4.0 \
#    --grid Hash \
#    --start 20 \
#    --view_select --vs_seed 37661 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/fewshot30/epoch=19.ckpt \
#    --start 30 --N_vs 6 --view_step 2 --epoch_step 20 \
#    --pre_train_epoch 20 \
#    --n_centers 10 \
#    --vs_by entropy \
#    --vs_sample_rate 1.0

######### mcd-r #######################

#python train_nsvf.py \
#    --root_dir $ROOT_DIR/${SCENES} --dataset_name splitnerfpp \
#    --exp_name ${SCENES}/fewshot42/ --eval_lpips \
#    --num_epochs 20 --scale 4.0 \
#    --grid Hash \
#    --start 20 \
#    --view_select --vs_seed 37661 \
#    --ckpt_path ${BASE_DIR}/${SCENES}/fewshot30/epoch=19.ckpt \
#    --start 30 --N_vs 6 --view_step 2 --epoch_step 20 \
#    --pre_train_epoch 20 \
#    --n_centers 10 \
#    --vs_by mcd_r --n_passes 10 --p 0.2 \
#    --vs_sample_rate 1.0

######### entropy #######################
python train_nsvf.py \
    --root_dir $ROOT_DIR/${SCENES} --dataset_name nerfpp \
    --exp_name ${SCENES}/fewshot62v2/ --eval_lpips \
    --num_epochs 20 --scale 4.0 \
    --grid Hash \
    --view_select --vs_seed 37661 \
    --ckpt_path ${BASE_DIR}/${SCENES}/fewshot50/epoch=19.ckpt \
    --start 50 --N_vs 6 --view_step 2 --epoch_step 20 \
    --pre_train_epoch 20 \
    --n_centers 10 \
    --vs_by entropy \
    --ray_sampling_strategy weighted_images \
    --vs_sample_rate 1.0


######### mcd-r #######################

python train_nsvf.py \
    --root_dir $ROOT_DIR/${SCENES} --dataset_name nerfpp \
    --exp_name ${SCENES}/fewshot62v2/ --eval_lpips \
    --num_epochs 20 --scale 4.0 \
    --grid Hash \
    --view_select --vs_seed 37661 \
    --ckpt_path ${BASE_DIR}/${SCENES}/fewshot50/epoch=19.ckpt \
    --start 50 --N_vs 6 --view_step 2 --epoch_step 20 \
    --pre_train_epoch 20 \
    --n_centers 10 \
    --ray_sampling_strategy weighted_images \
    --vs_by mcd_r --n_passes 10 --p 0.2 \
    --vs_sample_rate 1.0

