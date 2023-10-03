#!/bin/bash

export ROOT_DIR=/home/ubuntu/data/nerf_data/Synthetic_NeRF

### hash T20 64ch
python train.py \
    --root_dir $ROOT_DIR/Chair \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/Chair \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Drums \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/Drums \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2
    
python train.py \
    --root_dir $ROOT_DIR/Ficus \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/Ficus \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Hotdog \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/Hotdog \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/Lego \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Materials \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/Materials \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Mic \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/Mic \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Ship \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/Ship \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2



### hash T22 64ch
python train.py \
    --root_dir $ROOT_DIR/Chair \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/Chair \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Drums \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/Drums \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2
    
python train.py \
    --root_dir $ROOT_DIR/Ficus \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/Ficus \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Hotdog \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/Hotdog \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/Lego \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Materials \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/Materials \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Mic \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/Mic \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/Ship \
    --exp_name benchmark_20ksteps_batches_16384/Synthetic_NeRF/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/Ship \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

