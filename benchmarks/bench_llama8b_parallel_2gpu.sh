#!/bin/bash

# 定义模型路径变量，使代码更整洁
MODEL_PATH="/home/lishen/SpecForge/model/Llama-3.1-8B-Instruct"
DRAFT_MODEL_PATH="/home/lishen/SpecForge/outputs/llama3.1-8b-eagle3-perfectblend-llama3.1-8b/epoch_0_step_354683"

echo "启动 GPU 0 测评任务..."
CUDA_VISIBLE_DEVICES=0 python3 benchmarks/bench_eagle3.py \
    --model-path $MODEL_PATH \
    --speculative-draft-model-path $DRAFT_MODEL_PATH \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --attention-backend fa3 \
    --config-list 1,3,1,4 \
    --benchmark-list mtbench gpqa financeqa gsm8k \
    --dtype bfloat16 > benchmark_gpu0.log 2>&1 &

echo "启动 GPU 1 测评任务..."
CUDA_VISIBLE_DEVICES=1 python3 benchmarks/bench_eagle3.py \
    --model-path $MODEL_PATH \
    --speculative-draft-model-path $DRAFT_MODEL_PATH \
    --port 30001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --attention-backend fa3 \
    --config-list 1,3,1,4 \
    --benchmark-list math500 humaneval livecodebench \
    --dtype bfloat16 > benchmark_gpu1.log 2>&1 &

echo "两个任务均已在后台启动。"
echo "可以使用 'tail -f benchmark_gpu0.log' 或 'benchmark_gpu1.log' 查看实时进度。"
echo "等待所有任务执行完毕..."

# wait 命令会等待上述所有放入后台（&）的进程执行完毕
wait

echo "所有数据集测评已完成！"