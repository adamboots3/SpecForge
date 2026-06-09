#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
NUM_GPUS=${1:-2}

ATTENTION_BACKEND=${2:-sdpa}

LOG_DIR=$ROOT_DIR/logs
mkdir -p $LOG_DIR
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/qwen3_4b_dflash_${TS}.log
ln -sfn $LOG_FILE $LOG_DIR/qwen3_4b_dflash_latest.log
echo "[run_qwen3_4b_dflash_online] logging to $LOG_FILE"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path Qwen/Qwen3-4B \
    --draft-config-path $ROOT_DIR/configs/qwen3-4b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/gsm8k_qwen3-4b_regen.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-4b-gsm8k \
    --num-epochs 6 \
    --batch-size 1 \
    --tp-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 1024 \
    --chat-template qwen3-instruct \
    --attention-backend $ATTENTION_BACKEND \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to none \
    --wandb-project specforge-qwen3-4b-dflash \
    --target-model-backend sglang \
    --block-size 16 \
    --num-anchors 128 \
    --wandb-name qwen3-4b-dflash-perfectblend \
    --sglang-mem-fraction-static 0.34 \
    --resume \
    2>&1 | tee -a $LOG_FILE
