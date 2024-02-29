#!/bin/bash

export ROOT_DIR=/mnt/Data2/liyan/MF-NeRF/Data/
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nerfvs/
export CUDA_VISIBLE_DEVICES=0

scenes=(Hotdog Chair Drums Ficus)
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
    --exp_name ${SCENES}_${metric}_v2 \
    --loss bce \
    --num_epochs 300 --batch_size 128 --lr 1e-3 \
    --seed 79663

done