#!/bin/bash
# Regenerate gsm8k dataset for Qwen3-4B target model.
#
# Usage:
#   bash scripts/regen_qwen3_4b_gsm8k.sh [NUM_GPUS]
#
# NUM_GPUS defaults to 1. Pass 2 to launch two SGLang servers in parallel
# (one per GPU) and double throughput during regeneration.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Prevent corporate proxy from intercepting localhost requests during sglang warmup.
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export HF_HUB_OFFLINE=1

NUM_GPUS=${1:-1}
MODEL="Qwen/Qwen3-4B"
INPUT_JSONL="$ROOT_DIR/cache/dataset/gsm8k_train.jsonl"
OUTPUT_JSONL="$ROOT_DIR/cache/dataset/gsm8k_qwen3-4b_regen.jsonl"

BASE_PORT=30000

# ── Step 1: prepare seed data ──────────────────────────────────────────────────
if [[ -f "$INPUT_JSONL" ]]; then
    echo "[regen] Seed data already exists at $INPUT_JSONL, skipping prepare_data.py"
else
    echo "[regen] Preparing gsm8k seed data..."
    python "$ROOT_DIR/scripts/prepare_data.py" --dataset gsm8k
fi

# ── Step 2: launch SGLang server(s) ───────────────────────────────────────────
SERVER_PIDS=()
SERVER_ADDRESSES=()

launch_server() {
    local gpu_id=$1
    local port=$((BASE_PORT + gpu_id))
    echo "[regen] Launching SGLang server on GPU $gpu_id, port $port ..."
    CUDA_VISIBLE_DEVICES=$gpu_id python3 -m sglang.launch_server \
        --model "$MODEL" \
        --dtype bfloat16 \
        --mem-fraction-static 0.8 \
        --port "$port" \
        --cuda-graph-max-bs 128 \
        --host 127.0.0.1 &
    SERVER_PIDS+=($!)
    SERVER_ADDRESSES+=("localhost:$port")
}

for ((i = 0; i < NUM_GPUS; i++)); do
    launch_server "$i"
done

# Gracefully kill all servers on exit / Ctrl-C
cleanup() {
    echo "[regen] Stopping SGLang servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# Wait until all servers are ready
for addr in "${SERVER_ADDRESSES[@]}"; do
    host="${addr%%:*}"
    port="${addr##*:}"
    echo "[regen] Waiting for server at $addr ..."
    until curl -sf "http://$host:$port/health" >/dev/null 2>&1; do
        sleep 3
    done
    echo "[regen] Server at $addr is ready."
done

# ── Step 3: regenerate dataset ────────────────────────────────────────────────
echo "[regen] Starting dataset regeneration (NUM_GPUS=$NUM_GPUS)..."
python "$ROOT_DIR/scripts/regenerate_train_data.py" \
    --model "$MODEL" \
    --concurrency 128 \
    --max-tokens 2048 \
    --temperature 0.8 \
    --server-address "${SERVER_ADDRESSES[@]}" \
    --input-file-path "$INPUT_JSONL" \
    --output-file-path "$OUTPUT_JSONL" \
    --resume

echo "[regen] Done. Output: $OUTPUT_JSONL"
