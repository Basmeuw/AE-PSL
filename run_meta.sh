#!/bin/bash

"/Users/basmeuw/Dropbox/Education/MASTER/Y6 THESIS/Repository/AE-PSL/.venv/bin/python" "/Users/basmeuw/Dropbox/Education/MASTER/Y6 THESIS/Repository/AE-PSL/src/main_centralized.py"  \
    --model meta_transformer \
    --dataset cifar100 \
    --batch_size 500 \
    --nr_of_epochs 1 \
    --nr_of_last_encoder_blocks_to_finetune 6 \
    --start_lr 1e-4 \
    --random_seed 2024 \
    --save_file_name my_trained_model \
    --torch_data_dir data/torch \
    --pre_processors_cache_dir data/pre_processors_cache \
    --tokenizer_weights_cache_dir data/tokenizer_weights_cache \
    --model_weights_dir data/models