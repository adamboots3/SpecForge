#!/bin/bash
# Full pipeline: train EAGLE3 -> benchmark
# Regeneration skipped: ShareGPT avg 12-turn convs take ~155h to regen on 2x4090;
# EAGLE paper confirms model is not sensitive to regen, original data is sufficient.
# Auto-managed, unattended run. Logs go to ./logs/

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com

ROOT_DIR=/home/arda/li/SpecForge
LOG_DIR=$ROOT_DIR/logs
TRAIN_OUTPUT_DIR=$ROOT_DIR/outputs/qwen3-4b-eagle3-sharegpt_train

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"; }

log "=== STEP 1: Data Regeneration SKIPPED ==="
log "Reason: ShareGPT 120675 samples at ~13 samples/min = ~155h. Budget is 5 days."
log "Using original sharegpt_train.jsonl (EAGLE paper: not sensitive to regen quality)."

# ============================================================
# STEP 2: Train EAGLE3 (1 epoch)
# ============================================================
log "=== STEP 2: Training EAGLE3 ==="

TRAIN_DATA="$ROOT_DIR/cache/dataset/sharegpt_train.jsonl"
log "Training data: $TRAIN_DATA"

mkdir -p "$TRAIN_OUTPUT_DIR"

torchrun \
    --standalone \
    --nproc_per_node 2 \
    "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path Qwen/Qwen3-4B \
    --draft-model-config "$ROOT_DIR/configs/qwen3-4b-eagle3.json" \
    --train-data-path "$TRAIN_DATA" \
    --build-dataset-num-proc 64 \
    --output-dir "$TRAIN_OUTPUT_DIR" \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir "$ROOT_DIR/cache" \
    --embedding-key model.embed_tokens.weight \
    --tp-size 2 \
    --target-model-backend sglang \
    --attention-backend sdpa \
    --sglang-mem-fraction-static 0.2 \
    --resume \
    2>&1 | tee -a "$LOG_DIR/train.log"

log "Training complete."

# ============================================================
# STEP 3: Benchmark
# ============================================================
log "=== STEP 3: Benchmark ==="

# Find latest checkpoint
LATEST_CKPT=$(ls -td "$TRAIN_OUTPUT_DIR"/epoch_* 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    log "ERROR: No checkpoint found in $TRAIN_OUTPUT_DIR"
    exit 1
fi
log "Using checkpoint: $LATEST_CKPT"

python3 "$ROOT_DIR/benchmarks/bench_eagle3.py" \
    --model-path Qwen/Qwen3-4B \
    --speculative-draft-model-path "$LATEST_CKPT" \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.2 \
    --tp-size 2 \
    --attention-backend fa3 \
    --config-list 1,0,0,0 1,3,1,4 \
    --benchmark-list mtbench gsm8k:5 ceval:5:accountant \
    --dtype bfloat16 \
    --output-dir "$ROOT_DIR/results-sharegpt" \
    2>&1 | tee -a "$LOG_DIR/bench.log"

log "=== PIPELINE COMPLETE ==="
