#!/bin/bash

export LOG_DIR=./logs/nsvf/Synthetic_NSVF/Hash/fewshot30_v2/
export CUDA_VISIBLE_DEVICES=0

scenes=(Wineholder Robot Bike Spaceship Steamtrain)
#scenes=(Lifestyle Toad Palace)
for SCENES in ${scenes[@]}
do
python process_logs.py \
--log_dir ${LOG_DIR} \
--scenes Wineholder Robot Bike Spaceship Steamtrain \
--N_vs 4 \
--method random
done