#!/bin/bash

export PATH=/home/lishen/miniforge3/envs/li-te/bin:$PATH
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com

MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
PORT=30000

# Step 1: Start SGLang server
/home/lishen/miniforge3/envs/li-te/bin/python -m sglang.launch_server \
    --model ${MODEL} \
    --mem-fraction-static 0.8 \
    --tp 4 \
    --trust-remote-code \
    --cuda-graph-max-bs 128 \
    --host 127.0.0.1 \
    --port ${PORT} \
    --dtype bfloat16 &

SERVER_PID=$!
echo "SGLang server started with PID ${SERVER_PID}"

# Wait for server to be ready
echo "Waiting for server to be ready..."
until curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; do
    sleep 5
done
echo "Server is ready."

# Step 2: Regenerate dataset
cd /home/lishen/SpecForge && \
/home/lishen/miniforge3/envs/li-te/bin/python scripts/regenerate_train_data.py \
    --model ${MODEL} \
    --concurrency 64 \
    --max-tokens 4096 \
    --server-address 127.0.0.1:${PORT} \
    --temperature 0.8 \
    --resume \
    --input-file-path ./cache/dataset/gsm8k_train.jsonl \
    --output-file-path ./cache/dataset/gsm8k_train_regen_qwen80B.jsonl \
    2>&1 | tee /home/lishen/SpecForge/logs/gsm8k_train_regen_qwen80B.log

# Step 3: Stop server
kill ${SERVER_PID}
echo "Server stopped."
