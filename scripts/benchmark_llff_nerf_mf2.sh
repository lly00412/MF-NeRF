#!/bin/bash

export ROOT_DIR=/mnt/Data2/datasets/nerf_llff_data/
scenes=(fern flower fortress horns leaves orchids room trex)
for SCENES in ${scenes[@]}
do
echo ${SCENES}
### mfnerf T20 128ch
CUDA_VISIBLE_DEVICES=1 \
python train.py \
    --root_dir /mnt/Data2/datasets/nerf_llff_data/${SCENES} \
    --dataset_name colmap \
    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${SCENES} \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
    --rgb_channels 128 --rgb_layers 2
done