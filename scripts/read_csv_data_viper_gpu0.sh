#!/bin/bash

losses=l2
export BASE_DIR=/mnt/Data2/liyan/MF-NeRF/results/nsvf/Synthetic_NeRF/Hash/

python read_data_from_csv.py \
    --log_dir ${BASE_DIR}/fewshot11/ \
    --scenes Hotdog Chair Drums Ficus \
    --file_name eval_scores2 \
    --save_name eval_summary2