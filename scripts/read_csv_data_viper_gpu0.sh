#!/bin/bash

losses=l2
#export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/results/nsvf/Synthetic_NeRF/Hash/

#python read_data_from_csv.py \
#    --log_dir ${BASE_DIR}/fewshot11/ \
#    --scenes Hotdog Chair Drums Ficus \
#    --file_name eval_scores2 \
#    --save_name eval_summary2

export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/results/colmap/nerf_llff/Hash/res0.25/fewshot15_sr0.1
scenes=(room horns trex fortress)

for SCENES in ${scenes[@]}
do

python read_data_from_csv.py \
    --log_dir ${BASE_DIR}/${SCENES}/random/ \
    --file_name render \
    --save_name ${SCENES}_vs

python read_data_from_csv.py \
    --log_dir ${BASE_DIR}/${SCENES}/reweighted/entropy/ \
    --file_name render \
    --save_name ${SCENES}_vs

python read_data_from_csv.py \
    --log_dir ${BASE_DIR}/${SCENES}/reweighted/mcd_d/ \
    --file_name render \
    --save_name ${SCENES}_vs

python read_data_from_csv.py \
    --log_dir ${BASE_DIR}/${SCENES}/reweighted/mcd_r/ \
    --file_name render \
    --save_name ${SCENES}_vs

python read_data_from_csv.py \
    --log_dir ${BASE_DIR}/${SCENES}/reweighted/theta_3/warp/ \
    --file_name render \
    --save_name ${SCENES}_vs

done