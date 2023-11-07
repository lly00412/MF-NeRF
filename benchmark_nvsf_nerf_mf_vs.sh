#!/bin/bash

losses=l2
export ROOT_DIR=/mnt/Data2/datasets/Synthetic_NSVF/
export CKPT_DIR=/mnt/Data2/liyan/MF-NeRF/ckpts/nsvf/Synthetic_NSVF/Hash/${losses}/fewshot50/
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

scenes=(Wineholder)
#scenes=(Wineholder Steamtrain Toad Robot Bike Palace Spaceship Lifestyle)
sample_rate=(1.0)

fewshot_no=(50)

#### train the base line

#for SCENES in ${scenes[@]}
#do
#  echo ${SCENES}
#  for FS in ${fewshot_no}
#  do
#    echo train_on_${FS}_imgs
#  ### mfnerf T20 128ch
#    python train.py \
#      --root_dir ${ROOT_DIR}/${SCENES} \
#      --dataset_name nsvf \
#      --fewshot ${FS} --fewshot_seed 489 \
#      --exp_name Synthetic_NSVF/Hash/${losses}/fewshot${FS}/${SCENES}/ \
#      --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#      --rgb_channels 128 --rgb_layers 2 \
#      --loss ${losses}
#  done
#done
#

############### randon choice

for SCENES in ${scenes[@]}
do
  echo ${SCENES}
  for FS in ${fewshot_no}
  do
    echo train_on_${FS}_imgs
  ### mfnerf T20 128ch
    python train.py \
      --root_dir ${ROOT_DIR}/${SCENES} \
      --dataset_name nsvf \
      --fewshot ${FS} --fewshot_seed 489 \
      --exp_name Synthetic_NSVF/Hash/${losses}/fewshot${FS}/${SCENES}/ \
      --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
      --rgb_channels 128 --rgb_layers 2 \
      --loss ${losses} \
      --view_select --ckpt ${CKPT_DIR}/${SCENES}/epoch=19.ckpt \
      --pick_by random \
      --n_view 5 \
      --retrain
  done
done


################## pick by warp

#for FS in ${fewshot_no[@]}
#do
#for SR in ${sample_rate[@]}
#do
#for SCENES in ${scenes[@]}
#    do
#    echo ${SCENES}
#    echo sample_rate: ${SR}
#      ### sample by warp uncert
#    python train.py \
#        --root_dir ${ROOT_DIR}/${SCENES} \
#        --fewshot ${FS} --fewshot_seed 489 \
#        --dataset_name nsvf \
#        --exp_name Synthetic_NSVF/Hash/${losses}/fewshot${FS}/${SCENES}/ \
#        --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#        --rgb_channels 128 --rgb_layers 2 \
#        --loss ${losses} \
#        --view_select --ckpt ${CKPT_DIR}/${SCENES}/epoch=19.ckpt \
#        --vs_sample_rate ${SR} --vs_batch_size 2048 \
#        --pick_by warp \
#        --n_view 5 \
#        --retrain
#done
#done
#done

for SCENES in ${scenes[@]}
do
  for FS in ${fewshot_no}
  do
  for SR in ${sample_rate[@]}
  do
    echo ${SCENES}
    echo train_on_${FS}_imgs
    echo sample_rate: ${SR}
  ### mfnerf T20 128ch
    python train.py \
      --root_dir ${ROOT_DIR}/${SCENES} \
      --dataset_name nsvf \
      --fewshot ${FS} --fewshot_seed 489 \
      --exp_name Synthetic_NSVF/Hash/${losses}/fewshot${FS}/${SCENES}/ \
      --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
      --rgb_channels 128 --rgb_layers 2 \
      --loss ${losses} \
      --view_select --ckpt ${CKPT_DIR}/${SCENES}/epoch=19.ckpt \
      --vs_sample_rate ${SR} --vs_batch_size 2048 \
      --pick_by warp \
      --n_view 5 \
      --retrain
  done
done
done

################## pick by mc-dropout depth
#for SCENES in ${scenes[@]}
#do
#  for FS in ${fewshot_no}
#  do
#  for SR in ${sample_rate[@]}
#  do
#    echo ${SCENES}
#    echo train_on_${FS}_imgs
#    echo sample_rate: ${SR}
#  ### mfnerf T20 128ch
#    python train.py \
#      --root_dir ${ROOT_DIR}/${SCENES} \
#      --dataset_name nsvf \
#      --fewshot ${FS} --fewshot_seed 489 \
#      --exp_name Synthetic_NSVF/Hash/${losses}/fewshot${FS}/${SCENES}/ \
#      --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
#      --rgb_channels 128 --rgb_layers 2 \
#      --loss ${losses} \
#      --view_select --ckpt ${CKPT_DIR}/${SCENES}/epoch=19.ckpt \
#      --vs_sample_rate ${SR} --vs_batch_size 2048 \
#      --pick_by mcd --vals depth \
#      --n_passes 30 --p 0.2 \
#      --n_view 5 \
#      --retrain
#  done
#done
#done





#for SR in ${sample_rate[@]}
#  do
#    for SCENES in ${scenes[@]}
#      do
#      echo ${SCENES}
#      echo sample_rate: ${SR}
#
#      ### sample by warp uncert
#      python train.py \
#        --root_dir ${ROOT_DIR}/${SCENES} \
#        --fewshot 10 --fewshot_seed 399 \
#        --downsample 0.5 \
#        --dataset_name colmap \
#        --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/half_res/${SCENES}/ \
#        --num_epochs 20 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
#        --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
#        --rgb_channels 128 --rgb_layers 2 \
#        --loss ${losses} \
#        --view_select --ckpt ${CKPT_DIR}/half_res/${SCENES}/epoch=19.ckpt \
#        --vs_sample_rate ${SR} --vs_batch_size 1024 \
#        --pick_by warp \
#        --n_view 4 \
#        --retrain
#
#      ### sample by mcd depth
#      echo ${SCENES}
#      echo sample_rate: ${SR}
#
#      python train.py \
#        --root_dir ${ROOT_DIR}/${SCENES} \
#        --fewshot 10 --fewshot_seed 399 \
#        --downsample 0.5 \
#        --dataset_name colmap \
#        --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/half_res/${SCENES}/ \
#        --num_epochs 20 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
#        --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
#        --rgb_channels 128 --rgb_layers 2 \
#        --loss ${losses} \
#        --view_select --ckpt ${CKPT_DIR}/half_res/${SCENES}/epoch=19.ckpt \
#        --vs_sample_rate ${SR} --vs_batch_size 1024 \
#        --pick_by mcd --vals depth \
#        --n_passes 30 --p 0.2 \
#        --n_view 4 \
#        --retrain
#
#      ### sample by mcd rgb
#      echo ${SCENES}
#      echo sample_rate: ${SR}
#
#      python train.py \
#        --root_dir ${ROOT_DIR}/${SCENES} \
#        --fewshot 10 --fewshot_seed 399 \
#        --downsample 0.5 \
#        --dataset_name colmap \
#        --exp_name nerf_llff/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/${losses}/half_res/${SCENES}/ \
#        --num_epochs 20 --batch_size 2048 --scale 16.0 --lr 2e-2 --eval_lpips \
#        --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_tables 8 \
#        --rgb_channels 128 --rgb_layers 2 \
#        --loss ${losses} \
#        --view_select --ckpt ${CKPT_DIR}/half_res/${SCENES}/epoch=19.ckpt \
#        --vs_sample_rate ${SR} --vs_batch_size 1024 \
#        --pick_by mcd --vals rgb \
#        --n_passes 30 --p 0.2 \
#        --n_view 4 \
#        --retrain
#
#      done
#  done