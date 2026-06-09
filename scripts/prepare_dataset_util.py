import hashlib
import os
from datasets import load_dataset, Dataset
import random

# ==========================================
# 配置参数
# ==========================================
ORIGINAL_DATASET_ID = "mlabonne/open-perfectblend"
REGENERATED_DATASET_ID = "lukeysong/qwen3-80b-regen-prefectblend"

# 统一分层采样率：10%
SAMPLING_RATIO = 0.10
SEED = 42

# 输出路径：与项目数据集目录对齐
OUTPUT_PATH = "cache/dataset/perfectblend_qwen3-80b_regen_10pct.jsonl"

# Unknown 占比超过该阈值时发出警告（hash miss 过多说明 prompt 文本对不上）
UNKNOWN_WARN_THRESHOLD = 0.05

# ==========================================
# 辅助函数：提取第一轮 User 提问的 Hash 值
# ==========================================
def get_prompt_hash(messages):
    """提取对话中首个 User 提问计算 MD5，作为主键"""
    # 防止 messages 本身为 None
    if not messages:
        return None
        
    for msg in messages:
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value")
        
        if role in ["user", "human"]:
            # 【修复点】：防止 content 为 None 或非字符串格式导致报错
            if content is None:
                content = ""
            else:
                content = str(content) 
                
            clean_text = content.strip()
            return hashlib.md5(clean_text.encode('utf-8')).hexdigest()
            
    return None

def main():
    print("1. 正在加载原始数据集和重生成数据集...")
    orig_ds = load_dataset(ORIGINAL_DATASET_ID, split="train")
    regen_ds = load_dataset(REGENERATED_DATASET_ID, split="train")
    
    print(f"原始数据量: {len(orig_ds)} | 重生成数据量: {len(regen_ds)}")

    print("2. 正在构建 Hash -> Source 的映射索引...")
    hash_to_source = {}
    for row in orig_ds:
        msgs = row.get("messages") or row.get("conversations")
        h = get_prompt_hash(msgs)
        if h:
            hash_to_source[h] = row.get("source", "Unknown")

    print(f"成功建立 {len(hash_to_source)} 条哈希映射。")

    print("3. 正在将 Source 标签挂载到重生成数据集...")
    buckets = {}  # source -> list of rows
    for row in regen_ds:
        msgs = row.get("messages") or row.get("conversations")
        h = get_prompt_hash(msgs)
        source = hash_to_source.get(h, "Unknown") if h else "Unknown"
        row["source"] = source
        buckets.setdefault(source, []).append(row)

    total_regen = sum(len(v) for v in buckets.values())
    unknown_n = len(buckets.get("Unknown", []))
    unknown_ratio = unknown_n / total_regen if total_regen else 0
    print(f"Hash 映射命中率: {(1 - unknown_ratio) * 100:.2f}%  (Unknown: {unknown_n}/{total_regen})")
    if unknown_ratio > UNKNOWN_WARN_THRESHOLD:
        print(f"  ⚠️  Unknown 占比 {unknown_ratio * 100:.2f}% 超过阈值 {UNKNOWN_WARN_THRESHOLD * 100:.0f}%，"
              f"可能 prompt 文本格式不一致（空格/模板残留），请核查 get_prompt_hash 的规范化逻辑。")

    print("\n4. 执行真分层抽样：按 source 分桶后各取 10%...")
    rng = random.Random(SEED)
    sampled_data = []
    sampled_counter = {}
    for src, rows in buckets.items():
        # 至少留 1 条；不足 10 条的类别全保留以避免丢失稀有领域
        k = max(1, round(len(rows) * SAMPLING_RATIO))
        k = min(k, len(rows))
        picked = rng.sample(rows, k)
        sampled_data.extend(picked)
        sampled_counter[src] = k

    print("\n5. 抽样完成！分层对比报告：")
    print("-" * 80)
    print(f"{'Source (领域)':<40} | {'总数':<10} | {'应抽':<8} | {'实抽':<8} | {'实际比例'}")
    print("-" * 80)
    for src, rows in sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True):
        total = len(rows)
        expected = round(total * SAMPLING_RATIO)
        sampled_count = sampled_counter.get(src, 0)
        actual_ratio = (sampled_count / total) * 100 if total > 0 else 0
        print(f"{src[:38]:<40} | {total:<10} | {expected:<8} | {sampled_count:<8} | {actual_ratio:.2f}%")
    print("-" * 80)
    print(f"最终采出的 Draft Model 训练集大小: {len(sampled_data)} / {total_regen} "
          f"({len(sampled_data) / total_regen * 100:.2f}%)")

    # 6. 转换为 HuggingFace Dataset 并保存
    rng.shuffle(sampled_data)  # 打散，避免按 source 聚块
    final_ds = Dataset.from_list(sampled_data)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_ds.to_json(OUTPUT_PATH, force_ascii=False)
    print(f"处理完毕，已写入: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()