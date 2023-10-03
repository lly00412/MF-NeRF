#!/bin/bash

export ROOT_DIR=/home/ubuntu/data/nerf_data/360_v2
export DOWNSAMPLE=0.25 # to avoid OOM

### hash T20 64ch
python train.py \
    --root_dir $ROOT_DIR/bicycle --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/bicycle --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/bonsai --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/bonsai --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/counter --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/counter --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/garden --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/garden --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/kitchen --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/kitchen --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 4.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/room --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/room --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 4.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/stump --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/stump --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 64.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

# python train.py \
#     --root_dir $ROOT_DIR/flowers --dataset_name colmap \
#     --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/bicycle --downsample $DOWNSAMPLE \
#     --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#     --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#     --rgb_channels 64 --rgb_layers 2

# python train.py \
#     --root_dir $ROOT_DIR/treehill --dataset_name colmap \
#     --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T20_levels_16_F_2_rgb_2ly_64ch/bicycle --downsample $DOWNSAMPLE \
#     --num_epochs 20 --batch_size 4096 --scale 64.0 --lr 2e-2 --eval_lpips \
#     --L 16 --F 2 --T 20 --N_min 16 --grid Hash \
#     --rgb_channels 64 --rgb_layers 2



# ### hash T22 64ch
python train.py \
    --root_dir $ROOT_DIR/bicycle --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/bicycle --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/bonsai --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/bonsai --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/counter --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/counter --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/garden --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/garden --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/kitchen --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/kitchen --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 4.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/room --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/room --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 4.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

python train.py \
    --root_dir $ROOT_DIR/stump --dataset_name colmap \
    --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/stump --downsample $DOWNSAMPLE \
    --num_epochs 20 --batch_size 4096 --scale 64.0 --lr 2e-2 --eval_lpips \
    --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2

# python train.py \
#     --root_dir $ROOT_DIR/flowers --dataset_name colmap \
#     --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/bicycle --downsample $DOWNSAMPLE \
#     --num_epochs 20 --batch_size 4096 --scale 16.0 --lr 2e-2 --eval_lpips \
#     --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
#     --rgb_channels 64 --rgb_layers 2

# python train.py \
#     --root_dir $ROOT_DIR/treehill --dataset_name colmap \
#     --exp_name benchmark_20ksteps_batches_4096/mipnerf360/hashgrid_T22_levels_16_F_2_rgb_2ly_64ch/bicycle --downsample $DOWNSAMPLE \
#     --num_epochs 20 --batch_size 4096 --scale 64.0 --lr 2e-2 --eval_lpips \
#     --L 16 --F 2 --T 22 --N_min 16 --grid Hash \
#     --rgb_channels 64 --rgb_layers 2