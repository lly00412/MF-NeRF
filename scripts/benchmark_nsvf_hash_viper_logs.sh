#!/bin/bash
export LOG_DIR=./logs/nsvf/Synthetic_NSVF/Hash/fewshot15_sparse/
export CUDA_VISIBLE_DEVICES=0

#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Bike Lifestyle Robot Wineholder \
#--N_vs 4 \
#--method random
##
python process_logs_landa.py \
--log_dir ${LOG_DIR} \
--scenes Bike Lifestyle Wineholder \
--N_vs 4 \
--method reweighted/warp
##
##python process_logs_landa.py \
##--log_dir ${LOG_DIR} \
##--scenes Bike Lifestyle Robot Wineholder \
##--N_vs 5 \
##--method mcd_d
##
#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Bike Lifestyle Robot Wineholder \
#--N_vs 4 \
#--method reweighted/mcd_d
#
#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Bike Lifestyle Robot Wineholder \
#--N_vs 4 \
#--method reweighted/entropy

#python process_logs_landa.py \
#--log_dir ./logs/nsvf/Synthetic_NSVF/Hash/fewshot10 \
#--scenes Bike Lifestyle Robot Wineholder \
#--N_vs 1