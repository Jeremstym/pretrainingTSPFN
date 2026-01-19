#!/bin/bash

# Set OpenMP threads to avoid overhead
export OMP_NUM_THREADS=1

# Define your arguments in a variable for readability
TRAIN_ARGS="--nnodes=1 \
    --nproc_per_node=2 \
    /home/stympopper/pretrainingTSPFN/submodules/labram/run_class_finetuning.py \
    --output_dir /data/stympopper/LabramResults/ \
    --log_dir /data/stympopper/LabramResults/log/finetune_tuab_base \
    --model labram_base_patch200_200 \
    --finetune /home/stympopper/pretrainingTSPFN/ckpts/labram_vqnsp.pth \
    --weight_decay 0.05 \
    --batch_size 64 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --epochs 50 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --save_ckpt_freq 5 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset TUAB \
    --disable_qkv_bias \
    --seed 0"

# Execute using poetry run
poetry run torchrun $TRAIN_ARGS