#!/bin/bash

export ROOT_DIR=/home/ubuntu/data/nerf_data/Synthetic_NeRF

### mfnerf T20 128ch
CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Chair \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/Chair \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Drums \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/Drums \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2
    
CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Ficus \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/Ficus \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Hotdog \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/Hotdog \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/Lego \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Materials \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/Materials \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Mic \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/Mic \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Ship \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T20_levels_16_F_2_tables_8_rgb_2ly_128ch/Ship \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2



### mfnerf T22 128ch
CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Chair \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T22_levels_16_F_2_tables_8_rgb_2ly_128ch/Chair \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Drums \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T22_levels_16_F_2_tables_8_rgb_2ly_128ch/Drums \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2
    
CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Ficus \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T22_levels_16_F_2_tables_8_rgb_2ly_128ch/Ficus \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Hotdog \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T22_levels_16_F_2_tables_8_rgb_2ly_128ch/Hotdog \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T22_levels_16_F_2_tables_8_rgb_2ly_128ch/Lego \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Materials \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T22_levels_16_F_2_tables_8_rgb_2ly_128ch/Materials \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Mic \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T22_levels_16_F_2_tables_8_rgb_2ly_128ch/Mic \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --root_dir $ROOT_DIR/Ship \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/mfgrid_T22_levels_16_F_2_tables_8_rgb_2ly_128ch/Ship \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid MixedFeature --N_grids 8 \
    --rgb_channels 128 --rgb_layers 2