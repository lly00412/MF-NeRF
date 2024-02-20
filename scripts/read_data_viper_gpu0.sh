#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/nerf_datasets/Synthetic_NeRF/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/logs/nsvf/Synthetic_NeRF/Hash/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/logs/nsvf/Synthetic_NeRF/Hash/fewshot11/
export CUDA_VISIBLE_DEVICES=0
export PREFIX=Synthetic_NeRF/Hash/fewshot11

#export ROOT_DIR=/media/landa/lchen39/datasets/Synthetic_NeRF/
#export BASE_DIR=~/projects/MF-NeRF/ckpts/nsvf/Synthetic_NeRF/Hash/fewshot10/
#export CKPT_DIR=~/projects/MF-NeRF/ckpts/nsvf/Synthetic_NeRF/Hash/fewshot11/
#export CUDA_VISIBLE_DEVICES=0
#export PREFIX=Synthetic_NeRF/Hash/fewshot11/

#python read_data_from_logs.py \
#    --log_dir ${BASE_DIR}/fewshot11/ \
#    --scenes Hotdog \
#    --eval_u --u_by warp\
#    --N_vs 20 \
#    --v_num 2

#python read_data_from_logs.py \
#    --log_dir ${BASE_DIR}/fewshot10 \
#    --scenes Hotdog \
#    --eval_u --u_by warp\
#    --N_vs 1 \
#    --v_num 2

#python read_data_from_logs.py \
#    --log_dir ${BASE_DIR}/fewshot10 \
#    --scenes Chair Drums Ficus \
#    --eval_u --u_by warp\
#    --N_vs 1 \
#    --v_num 2

python read_data_from_logs.py \
    --log_dir ${BASE_DIR}/fewshot11/ \
    --scenes Chair Drums Ficus \
    --eval_u --u_by warp\
    --N_vs 20 \
    --v_num 2