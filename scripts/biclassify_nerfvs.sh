#!/bin/bash

export ROOT_DIR=/mnt/Data2/liyan/MF-NeRF/Data/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nerfvs/
export CUDA_VISIBLE_DEVICES=0

scenes=(Hotdog Chair Drums Ficus)
metric=lpips

for SCENES in ${scenes[@]}
do
echo ${SCENES}

python binarry_classifer_on_performace2.py \
    --data_file ${ROOT_DIR}/VS_Data.csv \
    --dataset_name nerfvs \
    --test_scene ${SCENES} \
    --target ${metric} \
    --exp_name test_on_${SCENES}_${metric}_v2 \
    --loss bce \
    --num_epochs 1000 --batch_size 64 --lr 1e-4 \
    --seed 79663

done