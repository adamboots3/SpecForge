#!/bin/bash

python3 benchmarks/bench_eagle3.py \
    --model-path Qwen/Qwen3-4B \
    --speculative-draft-model-path /home/arda/li/SpecForge/outputs/qwen3-4b-eagle3-sharegpt_train/epoch_0_step_60337 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.7 \
    --tp-size 2 \
    --attention-backend fa3 \
    --config-list 1,3,1,4 \
    --benchmark-list gpqa math500 humaneval \
    --dtype bfloat16