#!/bin/bash

#export LOG_DIR=./logs/nsvf/Synthetic_NSVF/Hash/fewshot20_v1/
#export CUDA_VISIBLE_DEVICES=0

#scenes=(Wineholder Robot Bike Spaceship Steamtrain)
#scenes=(Lifestyle Toad Palace)

#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Wineholder Robot Bike Spaceship Steamtrain Lifestyle Toad Palace \
#--N_vs 4 \
#--method random
#
#python process_logs.py \
#--log_dir ${LOG_DIR} \
#--scenes Chair Drums Ficus Hotdog \
#--N_vs 5 \
#--method warp
#
#python process_logs.py \
#--log_dir ${LOG_DIR} \
#--scenes Chair Drums Ficus Hotdog \
#--N_vs 5 \
#--method mcd_d
#
#python process_logs.py \
#--log_dir ${LOG_DIR} \
#--scenes Chair Drums Ficus Hotdog \
#--N_vs 5 \
#--method mcd_r

export LOG_DIR=./logs/nsvf/Synthetic_NSVF/Hash/fewshot20_sparse_v2/
export CUDA_VISIBLE_DEVICES=0

#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Chair Drums Ficus Hotdog \
#--N_vs 5 \
#--method random
#
#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Chair Drums Ficus Hotdog \
#--N_vs 5 \
#--method warp
#
python process_logs_landa.py \
--log_dir ${LOG_DIR} \
--scenes Chair Drums Ficus Hotdog \
--N_vs 5 \
--method mcd_d

python process_logs_landa.py \
--log_dir ${LOG_DIR} \
--scenes Chair Drums Ficus Hotdog \
--N_vs 5 \
--method mcd_r

#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Chair Drums Ficus Hotdog \
#--N_vs 5 \
#--method entropy