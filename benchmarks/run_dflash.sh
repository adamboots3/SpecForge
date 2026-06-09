#!/bin/bash
# Launch SGLang DFLASH server for the Qwen3-4B GSM8K-trained ckpt, then run dflash.benchmark
# across all five supported datasets. Server is killed on exit.
set -u

# ---- env ----
export http_proxy=http://child-prc.intel.com:913/
export https_proxy=http://child-prc.intel.com:913/
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export LD_LIBRARY_PATH=/home/arda/miniforge3/envs/dflash/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

PY=/home/arda/miniforge3/envs/dflash/bin/python
ROOT=/home/arda/li/SpecForge
DRAFT_CKPT=${DRAFT_CKPT:-$ROOT/outputs/qwen3-4b-gsm8k/epoch_2_step_8000}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
PORT=${PORT:-30000}
TP=${TP:-2}
NUM_PROMPTS=${NUM_PROMPTS:-128}
CONCURRENCY=${CONCURRENCY:-1}
DATASET_LIST=${DATASET_LIST:-gsm8k,math500,humaneval,mbpp,mt-bench}

LOG_DIR=$ROOT/logs/dflash_serve
RESULT_DIR=$ROOT/results
mkdir -p $LOG_DIR $RESULT_DIR
TS=$(date +%Y%m%d_%H%M%S)
SERVER_LOG=$LOG_DIR/server_${TS}.log
BENCH_LOG=$LOG_DIR/bench_all_${TS}.log
RESULT_FILE=$RESULT_DIR/results_dflash_${TS}.jsonl
ln -sfn $SERVER_LOG $LOG_DIR/server_latest.log
ln -sfn $BENCH_LOG  $LOG_DIR/bench_latest.log

echo "[run_dflash] server log:  $SERVER_LOG"
echo "[run_dflash] bench  log:  $BENCH_LOG"
echo "[run_dflash] result file: $RESULT_FILE"
echo "[run_dflash] draft ckpt:  $DRAFT_CKPT"

# ---- launch server ----
$PY -m sglang.launch_server \
    --model-path $TARGET_MODEL \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path $DRAFT_CKPT \
    --speculative-num-draft-tokens 16 \
    --tp-size $TP \
    --attention-backend fa3 \
    --speculative-draft-attention-backend fa3 \
    --mem-fraction-static 0.75 \
    --dtype bfloat16 \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port $PORT > $SERVER_LOG 2>&1 &
SERVER_PID=$!
echo "[run_dflash] server pid: $SERVER_PID"

cleanup() {
    echo "[run_dflash] stopping server $SERVER_PID ..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- wait for server ready ----
echo "[run_dflash] waiting for server on :$PORT ..."
for i in $(seq 1 120); do
    if grep -q "The server is fired up and ready to roll" $SERVER_LOG 2>/dev/null; then
        echo "[run_dflash] server ready after ${i}*10s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[run_dflash] ERROR: server died, see $SERVER_LOG"
        tail -40 $SERVER_LOG
        exit 1
    fi
    sleep 10
done

# ---- benchmark ----
cd $ROOT/dflash
$PY -m dflash.benchmark \
    --backend sglang \
    --base-url http://127.0.0.1:$PORT \
    --model $TARGET_MODEL \
    --dataset-list $DATASET_LIST \
    --num-prompts $NUM_PROMPTS \
    --concurrency $CONCURRENCY \
    --temperature 0.0 \
    --draft-model-path $DRAFT_CKPT \
    --output-file $RESULT_FILE \
    2>&1 | tee $BENCH_LOG

echo "[run_dflash] benchmark done. summary blocks:"
grep -E "Dataset:|Throughput:|Accept length:|Latency:|Output tokens:" $BENCH_LOG
