from datasets import load_dataset

def extract_pure_math_data():
    print("正在加载数据集 mlabonne/open-perfectblend (使用本地缓存)...")
    dataset = load_dataset("mlabonne/open-perfectblend", split="train")
    
    print(f"原始数据集总大小: {len(dataset)} 条")

    # 修正了 Orca 数据集的实际 source 字段名称
    target_sources = [
        "meta-math/MetaMathQA",
        "HuggingFaceH4/orca-math-word-problems-200k",  # 实际存在于数据集中的命名
        "microsoft/orca-math-word-problems-200k"       # 保留作为 fallback
    ]

    def is_exact_math_source(example):
        return example.get("source") in target_sources

    print("正在精确过滤指定的两个数学数据集...")
    pure_math_dataset = dataset.filter(is_exact_math_source, num_proc=8)

    print(f"提取完成！共提取到 {len(pure_math_dataset)} 条纯数学数据。")
    
    # 验证一下各个子数据集的具体数量
    from collections import Counter
    source_counts = Counter(pure_math_dataset['source'])
    print("\n--- 数据来源统计 ---")
    for src, count in source_counts.items():
        print(f"{src}: {count} 条")
    print("--------------------\n")

    output_file = "pure_math_only.jsonl"
    print(f"正在保存数据到 {output_file} ...")
    pure_math_dataset.to_json(output_file, force_ascii=False)
    print("保存成功！")

if __name__ == "__main__":
    extract_pure_math_data()