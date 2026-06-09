#!/bin/bash

export PATH=/home/lishen/miniforge3/envs/li-te/bin:$PATH
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export HF_ENDPOINT=https://hf-mirror.com

MODEL=/home/lishen/SpecForge/model/Llama-3.1-8B-Instruct
PORT0=30000
PORT1=30001

# Step 1: Start two independent SGLang servers (one per GPU, TP=1)
CUDA_VISIBLE_DEVICES=0 /home/lishen/miniforge3/envs/li-te/bin/python -m sglang.launch_server \
    --model ${MODEL} \
    --mem-fraction-static 0.85 \
    --tp 1 \
    --trust-remote-code \
    --cuda-graph-max-bs 128 \
    --host 127.0.0.1 \
    --port ${PORT0} \
    --dtype bfloat16 &

SERVER0_PID=$!
echo "SGLang server 0 started on GPU 0, PID ${SERVER0_PID}, port ${PORT0}"

CUDA_VISIBLE_DEVICES=1 /home/lishen/miniforge3/envs/li-te/bin/python -m sglang.launch_server \
    --model ${MODEL} \
    --mem-fraction-static 0.85 \
    --tp 1 \
    --trust-remote-code \
    --cuda-graph-max-bs 128 \
    --host 127.0.0.1 \
    --port ${PORT1} \
    --dtype bfloat16 &

SERVER1_PID=$!
echo "SGLang server 1 started on GPU 1, PID ${SERVER1_PID}, port ${PORT1}"

# Wait for both servers to be ready
echo "Waiting for both servers to be ready..."
until curl -s http://127.0.0.1:${PORT0}/health > /dev/null 2>&1; do sleep 5; done
echo "Server 0 (port ${PORT0}) is ready."
until curl -s http://127.0.0.1:${PORT1}/health > /dev/null 2>&1; do sleep 5; done
echo "Server 1 (port ${PORT1}) is ready."

# Step 2: Regenerate dataset using both servers (64 concurrency each, 128 total)
cd /home/lishen/SpecForge && \
/home/lishen/miniforge3/envs/li-te/bin/python scripts/regenerate_train_data_async.py \
    --model ${MODEL} \
    --concurrency 64 \
    --max-tokens 4096 \
    --server-address 127.0.0.1:${PORT0} 127.0.0.1:${PORT1} \
    --temperature 0.8 \
    --resume \
    --input-file-path ./cache/dataset/perfectblend_train_10pct_openai.jsonl \
    --output-file-path ./cache/dataset/perfectblend_train_10pct_openai_regen_llama8B.jsonl \
    2>&1 | tee /home/lishen/SpecForge/logs/regen_perfectblend_10pct_llama8B.log

# Step 3: Stop both servers
kill ${SERVER0_PID} ${SERVER1_PID}
echo "Both servers stopped."
