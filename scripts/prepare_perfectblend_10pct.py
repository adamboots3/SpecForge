import os
from datasets import load_dataset, concatenate_datasets

output_file = "cache/dataset/perfectblend_train_10pct.jsonl"

print("正在从 Hugging Face 云端加载完整数据集（包含 source 字段）...")
# 直接从 HF 加载
full_dataset = load_dataset("mlabonne/open-perfectblend", split="train")

sources = full_dataset.unique("source")
print(f"发现以下数据来源: {sources}")

sampled_subsets = []
sampling_ratio = 0.10

for source in sources:
    subset = full_dataset.filter(lambda x: x["source"] == source)
    sample_size = int(len(subset) * sampling_ratio)
    sampled_subset = subset.shuffle(seed=42).select(range(sample_size))
    sampled_subsets.append(sampled_subset)
    print(f"[{source}]: {len(subset)} -> {len(sampled_subset)}")

mini_perfectblend = concatenate_datasets(sampled_subsets).shuffle(seed=42)

print(f"\n采样完成！总大小: {len(mini_perfectblend)}")

# 考虑到你本地训练只需要 id 和 conversations，我们可以在保存前把多余的字段删掉
mini_perfectblend = mini_perfectblend.remove_columns(["source"])

os.makedirs(os.path.dirname(output_file), exist_ok=True)
mini_perfectblend.to_json(output_file, force_ascii=False)
print(f"已保存至: {output_file}")