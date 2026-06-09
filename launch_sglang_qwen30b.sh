#!/bin/bash
export PATH=/home/lishen/miniforge3/envs/li-te/bin:$PATH
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com

/home/lishen/miniforge3/envs/li-te/bin/python -m sglang.launch_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --mem-fraction-static 0.4 \
    --tp 4 \
    --trust-remote-code \
    --cuda-graph-max-bs 128 \
    --host 127.0.0.1 \
    --port 30000 \
    --dtype bfloat16 \


# cd /home/lishen/SpecForge && \
# /home/lishen/miniforge3/envs/li-te/bin/python scripts/regenerate_train_data.py \
#     --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --concurrency 64 \
#     --max-tokens 4096 \
#     --server-address 127.0.0.1:30000 \
#     --temperature 0.8 \
#     --resume \
#     --input-file-path ./cache/dataset/gsm8k_train.jsonl \
#     --output-file-path ./cache/dataset/gsm8k_train_regen.jsonl \
#     2>&1 | tee /home/lishen/SpecForge/regen_data.log