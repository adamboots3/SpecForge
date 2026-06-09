import json
from datasets import load_dataset

def extract_regenerated_math():
    # 1. 构建指纹库 (从刚才生成的纯数学数据中提取用户问题)
    print("正在从 pure_math_only.jsonl 构建问题匹配指纹库...")
    math_prompts = set()
    
    try:
        with open("pure_math_only.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 原版数据集的对话字段叫 'conversations'
                conversations = data.get("conversations", [])
                for msg in conversations:
                    # 原版的角色叫 'from'='human'，内容叫 'value'
                    if msg.get("from") == "human" or msg.get("role") == "user":
                        # 兼容 'value' (原版) 和 'content' 两种格式
                        content = msg.get("value") or msg.get("content", "")
                        math_prompts.add(content.strip())
                        break # 只需要第一轮的用户提问即可
    except FileNotFoundError:
        print("错误：找不到 pure_math_only.jsonl，请确保该文件在当前目录下。")
        return

    print(f"指纹库构建完成！共记录了 {len(math_prompts)} 个唯一的数学问题。")

    # 如果指纹库还是0，就不要往下跑了
    if len(math_prompts) == 0:
        print("提取失败：未能从 pure_math_only.jsonl 中读取到问题，请检查文件格式。")
        return

    # 2. 加载新数据集
    print("\n正在加载 frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct ...")
    new_dataset = load_dataset("frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct", split="train")
    print(f"新数据集总大小: {len(new_dataset)} 条")

    # 3. 过滤函数：检查新数据的问题是否在我们的指纹库中
    def is_math_in_regenerated(example):
        conversations = example.get("conversations", [])
        if not conversations:
            return False
        
        # 新数据集(frankleeeee)的格式是 "role": "user", "content": "..."
        for msg in conversations:
            if msg.get("role") == "user":
                user_content = msg.get("content", "").strip()
                return user_content in math_prompts
        return False

    # 4. 执行多进程过滤
    print("\n正在通过指纹比对，提取重新生成的数学数据 (多进程加速中)...")
    regenerated_math_ds = new_dataset.filter(is_math_in_regenerated, num_proc=8)

    print(f"\n提取完成！共匹配到 {len(regenerated_math_ds)} 条重新生成的纯数学数据。")

    # 5. 保存结果
    output_file = "regenerated_math_llama3.1_8b.jsonl"
    print(f"正在保存数据到 {output_file} ...")
    regenerated_math_ds.to_json(output_file, force_ascii=False)
    print("保存成功！")

if __name__ == "__main__":
    extract_regenerated_math()