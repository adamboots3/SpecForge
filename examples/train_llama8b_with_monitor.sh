#!/bin/bash

# ── Environment ──────────────────────────────────────────────────────────────
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/train_${TIMESTAMP}.log"
GPU_LOG="$LOG_DIR/gpu_mem_${TIMESTAMP}.csv"

# ── Conda environment ────────────────────────────────────────────────────────
source /home/lishen/miniforge3/etc/profile.d/conda.sh
conda activate li-te

# ── GPU memory monitor (background) ─────────────────────────────────────────
echo "timestamp,gpu_id,name,mem_used_MiB,mem_total_MiB,gpu_util_pct,power_W" > "$GPU_LOG"

nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits \
    --loop=30 \
    -i "$CUDA_VISIBLE_DEVICES" >> "$GPU_LOG" &
GPU_MON_PID=$!
echo "$GPU_MON_PID" > "$LOG_DIR/gpu_monitor_${TIMESTAMP}.pid"

echo "[launch] Training log : $TRAIN_LOG"
echo "[launch] GPU mem log  : $GPU_LOG"
echo "[launch] GPU monitor PID: $GPU_MON_PID"

# ── Training ─────────────────────────────────────────────────────────────────
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"

NUM_GPUS=2
TP_SIZE=1
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /home/lishen/SpecForge/model/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend-llama3.1-8b-instruct_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/llama3.1-8b-eagle3-perfectblend-llama3.1-8b \
    --num-epochs 3 \
    --batch-size 2 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 1024 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --target-model-backend sglang \
    --log-interval 10 \
    --sglang-mem-fraction-static 0.4 \
    --resume \
    > "$TRAIN_LOG" 2>&1

TRAIN_EXIT=$?

# ── Cleanup monitor ──────────────────────────────────────────────────────────
kill "$GPU_MON_PID" 2>/dev/null
echo "[launch] Training finished with exit code $TRAIN_EXIT"
exit $TRAIN_EXIT
