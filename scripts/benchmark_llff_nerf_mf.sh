#!/bin/bash

export ROOT_DIR=/mnt/Data2/datasets/nerf_llff_data/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/fewshot
export CUDA_VISIBLE_DEVICES=0
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
scenes=(fern flower fortress horns leaves orchids room trex)
for SCENES in ${scenes[@]}
do
echo ${SCENES}
### mfnerf T20 128ch
python train.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --dataset_name colmap \
    --fewshot 10 \
    --fewshot_seed 366 \
    --exp_name nerf_llff/mfgrid_T21_levels_16_F_2_tables_8_rgb_2ly_128ch/fewshot/${SCENES}/mcd100/ \
    --num_epochs 5 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
    --rgb_channels 128 --rgb_layers 2 \
    --mcdropout --n_passes 30 --p 0.5 \
    --val_only --ckpt_path ${CKPT_DIR}/${SCENES}/epoch=9.ckpt \
    --save_output
done

#--val_only --ckpt_path ${CKPT_DIR}/${SCENES}/epoch=19-val.ckpt \
#    --mcdropout --n_passes 10 --p 0.5

#