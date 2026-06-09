#!/bin/bash

python3 benchmarks/bench_eagle3.py \
    --model-path Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --speculative-draft-model-path /home/lishen/SpecForge/outputs/qwen3-80b-regen-gsm8k/epoch_2_step_5604 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.6 \
    --tp-size 4 \
    --attention-backend fa3 \
    --config-list 1,3,1,4 \
    --benchmark-list mtbench gpqa financeqa gsm8k math500 humaneval livecodebench \
    --dtype bfloat16