#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/datasets/nerf_llff_data/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/colmap/nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/
export CUDA_VISIBLE_DEVICES=0

scenes=(trex horns room)

#for SCENES in ${scenes[@]}
#do
#  echo ${SCENES}
#  ### mfnerf T20 128ch
#  python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --fewshot 10 --fewshot_seed 399 \
#    --downsample 0.5 \
#    --dataset_name colmap \
#    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/half_res/${SCENES}/ \
#    --num_epochs 20 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
#    --rgb_channels 128 --rgb_layers 2 \
#    --loss ${losses}
#  done

for SCENES in ${scenes[@]}
do
  echo ${SCENES}
  ### mfnerf T20 128ch
  python train.py \
    --root_dir ${ROOT_DIR}/${SCENES} \
    --fewshot 10 --fewshot_seed 399 \
    --downsample 0.5 \
    --dataset_name colmap \
    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/half_res/${SCENES}/ \
    --num_epochs 20 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
    --rgb_channels 128 --rgb_layers 2 \
    --loss ${losses} \
    --view_select --ckpt ${CKPT_DIR}/half_res/${SCENES}/epoch=19.ckpt \
    --pick_by warp \
    --n_view 4 \
    --retrain
  done

#for SCENES in ${scenes[@]}
#do
#  echo ${SCENES}
#  ### mfnerf T20 128ch
#  python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --fewshot 10 --fewshot_seed 399 \
#    --downsample 0.5 \
#    --dataset_name colmap \
#    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/half_res/${SCENES}/ \
#    --num_epochs 20 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
#    --rgb_channels 128 --rgb_layers 2 \
#    --loss ${losses} \
#    --view_select --ckpt ${CKPT_DIR}/half_res/${SCENES}/epoch=19.ckpt \
#    --pick_by mcd \
#    --n_passes 30 --p 0.2 \
#    --n_view 4 \
#    --retrain
#  done

#for SCENES in ${scenes[@]}
#do
#  echo ${SCENES}
#  ### mfnerf T20 128ch
#  python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --fewshot 10 --fewshot_seed 399 \
#    --downsample 0.5 \
#    --dataset_name colmap \
#    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/half_res/${SCENES}/ \
#    --num_epochs 20 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
#    --rgb_channels 128 --rgb_layers 2 \
#    --loss ${losses} \
#    --view_select --ckpt ${CKPT_DIR}/half_res/${SCENES}/epoch=19.ckpt \
#    --pick_by random \
#    --n_view 4 \
#    --retrain
#  done

#
#for SCENES in ${scenes[@]}
#do
#  echo ${SCENES}
#  ### mfnerf T20 128ch
#  export CUDA_LAUNCH_BLOCKING=1
#  export TORCH_USE_CUDA_DSA=1
#  python train.py \
#    --root_dir ${ROOT_DIR}/${SCENES} \
#    --downsample 0.5 \
#    --dataset_name colmap \
#    --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/norm_cams/half_res/${SCENES}/ \
#    --num_epochs 20 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
#    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
#    --rgb_channels 128 --rgb_layers 2 \
#    --loss ${losses} \
#    --mcdropout --n_passes 30 --p 0.2 \
#    --val_only --ckpt ${CKPT_DIR}/norm_cams/half_res/${SCENES}/epoch=19.ckpt \
#    --render_vcam \
#    --plot_roc
#  done
#
#    --mcdropout --n_passes 30 --p 0.2 \
#    --warp --ref_cam 0 \
#    --save_output \
#    --plot_roc