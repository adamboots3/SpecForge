#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=4
TP_SIZE=4
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --draft-model-config $ROOT_DIR/configs/qwen3-next-80b-a3b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_qwen3-80b_regen_10pct.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-80b-regen-blend \
    --num-epochs 3 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --target-model-backend sglang \
    --attention-backend sdpa \
    --sglang-mem-fraction-static 0.6
