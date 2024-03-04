#!/bin/bash
export LOG_DIR=./logs/nsvf/Synthetic_NSVF/Hash/fewshot15_v2/
export CUDA_VISIBLE_DEVICES=0

#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Robot \
#--N_vs 4 \
#--version 0 \
#--method random

#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Wineholder \
#--N_vs 4 \
#--version 1 \
#--method random
##
#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Robot \
#--N_vs 4 \
#--version 0 \
#--method reweighted/warp
#
python process_logs_landa.py \
--log_dir ${LOG_DIR} \
--scenes Robot \
--N_vs 4 \
--version 1 \
--method reweighted/entropy
#
#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Robot \
#--N_vs 4 \
#--version 0 \
#--method reweighted/10ps/mcd_d
#
#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Robot \
#--N_vs 4 \
#--version 0 \
#--method reweighted/30ps/mcd_d
#
#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Robot \
#--N_vs 4 \
#--version 0 \
#--method reweighted/10ps/mcd_r
#
#python process_logs_landa.py \
#--log_dir ${LOG_DIR} \
#--scenes Robot \
#--N_vs 4 \
#--version 0 \
#--method reweighted/30ps/mcd_r

