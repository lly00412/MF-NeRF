#!/bin/bash
export LOG_DIR=./logs/colmap/nerf_llff/Hash/res0.25/fewshot15_sparse/
export CUDA_VISIBLE_DEVICES=1

python process_logs_landa.py \
--log_dir ${LOG_DIR} \
--scenes room horns trex fortress \
--N_vs 4 \
--method random
##
python process_logs_landa.py \
--log_dir ${LOG_DIR} \
--scenes room horns trex fortress \
--N_vs 4 \
--method reweighted/theta_3/warp
##
python process_logs_landa.py \
--log_dir ${LOG_DIR} \
--scenes room horns trex fortress \
--N_vs 4 \
--method reweighted/mcd_d

python process_logs_landa.py \
--log_dir ${LOG_DIR} \
--scenes room horns trex fortress \
--N_vs 4 \
--method reweighted/mcd_r

python process_logs_landa.py \
--log_dir ${LOG_DIR} \
--scenes room horns trex fortress \
--N_vs 4 \
--method reweighted/entropy