#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=${1:-2}

ATTENTION_BACKEND=${2:-sdpa}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path /home/arda/li/SpecForge/model/Llama-3.1-8B-Instruct \
    --draft-config-path $ROOT_DIR/configs/llama3.1-8b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend-llama3.1-8b-instruct_train.jsonl \
    --output-dir $ROOT_DIR/outputs/llama3.1-8b-dflash-perfectblend \
    --num-epochs 6 \
    --batch-size 1 \
    --tp-size 2 \
    --mask-token-id 128002 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 1024 \
    --chat-template llama3 \
    --attention-backend $ATTENTION_BACKEND \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to none \
    --wandb-project specforge-llama3.1-8b-dflash \
    --target-model-backend sglang \
    --block-size 16 \
    --num-anchors 512 \
    --wandb-name llama3.1-8b-dflash-perfectblend \
    --sglang-mem-fraction-static 0.35
