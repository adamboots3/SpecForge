#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# 获取脚本所在目录及项目根目录
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# 硬件配置
NUM_GPUS=2
TP_SIZE=2
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# 启动训练
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-4B \
    --draft-model-config $ROOT_DIR/configs/qwen3-4b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/gsm8k_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-4b-eagle3-gsm8k_train \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 1024 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --attention-backend sdpa \
    --sglang-mem-fraction-static 0.55