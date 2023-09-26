#!/bin/bash

export ROOT_DIR=/mnt/Data2/datasets/nerf_llff_data/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts
#scenes=(fern flower fortress horns leaves orchids room trex)
scenes=(trex)
for SCENES in ${scenes[@]}
do
echo ${SCENES}
### mfnerf T20 128ch
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${SCENES} \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
    --rgb_channels 128 --rgb_layers 2 \
    --val_only \
    --mcdropout --n_passes 10
    #--val_only --ckpt_path ${CKPT_DIR}/colmap/nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${SCENES}/epoch=19-v2.ckpt \
done