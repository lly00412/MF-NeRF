#!/bin/bash

export LOG_DIR=./logs/nsvf/Synthetic_NSVF/Hash/fewshot30_v2/
export CUDA_VISIBLE_DEVICES=0

#scenes=(Wineholder Robot Bike Spaceship Steamtrain)
scenes=(Lifestyle Toad Palace)

python process_logs.py \
--log_dir ${LOG_DIR} \
--scenes Wineholder Robot Bike Spaceship Steamtrain Lifestyle Toad Palace \
--N_vs 4 \
--method random
#
python process_logs.py \
--log_dir ${LOG_DIR} \
--scenes Wineholder Robot Bike Spaceship Steamtrain Lifestyle Toad Palace \
--N_vs 4 \
--method reweighted/theta_3/warp

python process_logs.py \
--log_dir ${LOG_DIR} \
--scenes Wineholder Robot Bike Spaceship Steamtrain Lifestyle Toad Palace \
--N_vs 4 \
--method reweighted/mcd_d

python process_logs.py \
--log_dir ${LOG_DIR} \
--scenes Wineholder Robot Bike Spaceship Steamtrain Lifestyle Toad Palace \
--N_vs 4 \
--method reweighted/mcd_r
