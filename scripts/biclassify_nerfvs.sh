#!/bin/bash

export ROOT_DIR=/mnt/Data2/liyan/MF-NeRF/Data/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nerfvs/
export CUDA_VISIBLE_DEVICES=0

scenes=(Hotdog Chair Drums Ficus Lego Materials Mic Ship)
#scenes=(Lego Materials Mic Ship)
metric=psnr

for SCENES in ${scenes[@]}
do
echo ${SCENES}

python binarry_classifer_on_performace2.py \
    --data_dir ${ROOT_DIR} \
    --dataset_name nerfvs \
    --test_scene ${SCENES} \
    --target ${metric} \
    --exp_name ${SCENES}_${metric}_v3 \
    --loss bce \
    --num_epochs 300 --batch_size 2048 --lr 1e-4 \
    --seed 79663 \
    --u_bins 10

done

#export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/results/nsvf/Synthetic_NeRF/Hash/
#
#python read_data_from_csv.py \
#    --log_dir ${BASE_DIR}/fewshot11/ \
#    --scenes Lego Mic Materials Ship