#!/bin/bash

python3 benchmarks/bench_eagle3.py \
    --model-path /home/lishen/SpecForge/model/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path /home/lishen/SpecForge/outputs/llama3-8b-eagle3-perfectblend_train_10pct_openai_regen_llama8B/epoch_2_step_210000 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.6 \
    --tp-size 2 \
    --attention-backend fa3 \
    --config-list 1,3,1,4 \
    --benchmark-list mtbench gpqa financeqa gsm8k math500 humaneval livecodebench \
    --dtype bfloat16