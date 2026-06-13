import json
import os
from datetime import datetime


def update_sft_prompts(input_file, output_prefix):
    # 1. 获取当前时间戳用于命名：1.19.20:15

    output_file = "data/output_prefix.jsonl"

    # 2. 定义你想要的新 Prompt
    # 注意：对于多模态模型，通常需要保留 <image> 标签
    new_user_prompt = "这是城市发展对比图。请先识别新增地物的分布特征,并据此给出城市治理建议"

    updated_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            data = json.loads(line.strip())

            # 3. 修改 ID (可选，加上时间戳前缀)
            # data['id'] = f"{current_time}--{data['id']}"

            # 4. 找到 user 的对话块并修改 content
            # 假设 conversations[0] 是 user，[1] 是 assistant
            for msg in data['messages']:
                if msg['role'] == 'user':
                    # 保留原有的 <image> 标签，只替换文字内容
                    # 统计原有的 <image> 数量
                    msg['content'] = f"<image>\n<image>\n{new_user_prompt}"

            # 5. 写入新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            f_out.flush()  # 确保实时写入
            updated_count += 1

    print(f"转换完成！")
    print(f"原始文件: {input_file}")
    print(f"新文件: {output_file}")
    print(f"共处理数据: {updated_count} 条")


# --- 执行脚本 ---
# 假设你原来的文件叫 LEVIR-CD+_raw.jsonl
update_sft_prompts('01.19.16:12_LEVIR-CD+_sft.jsonl', 'LEVIR-CD+_refined')