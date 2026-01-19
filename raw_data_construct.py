import base64
import json
import os
import re
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

# 配置部分
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

model_name = "Qwen/Qwen3-VL-32B-Instruct"
# 根据你的图片路径修改
dataset_root = "dataset/LEVIR-CD+/train"
# 获取当前时间并格式化为: 月.日.时:分

current_time = datetime.now().strftime("%m.%d.%H:%M")
output_file = f"{current_time}_LEVIR-CD+_sft.jsonl"

prompt_text = """
你是一位精通遥感影像分析与城市规划规划的专家。请对比提供的两张时序影像（图A为前期，图B为后期），按照以下逻辑进行深度分析：

### 第一步：客观变化描述 (Objective Description)
1. 识别新增地物的类型（如：矩阵式住宅、单体建筑、工业厂房、硬化道路、绿化带）。
2. 观察变化的几何特征（如：布局是否整齐、是否沿交通线分布、建筑密度高低）。

### 第二步：开发模式识别 (Pattern Recognition)
基于视觉特征判定开发性质，禁止直接假设所有开发均为“无序”：
- **有序规划型**：建筑排列规整、道路网同步完善、伴随有公共绿化或配套设施。这通常反映了【存量土地优化】或【高水平城镇化进程】。
- **自发蔓延型**：建筑散落在林地/耕地边缘、缺乏系统性道路连接、侵占自然生态廊道。这可能反映了【无序扩张】或【监管盲区】。

### 第三步：多维治理评估 (Governance Assessment)
请根据以上判定，从两个对立维度进行评价：
1. **正面评价**：分析该开发如何提升了土地利用率（Land Use Efficiency）、改善了居住条件或促进了区域经济。
2. **潜在风险**：分析该变化对生态红线、热岛效应或公共服务压力带来的挑战。

### 第四步：精准政策建议 (Policy Recommendations)
- 如果是**有序开发**：建议侧重于“精细化运营”，如智慧社区建设、能源监测、交通动态优化。
- 如果是**无序开发**：建议侧重于“规划红线执法”，如遥感常态化监测、违建拆除、生态修复补偿。

---
请开始你的分析，保持专业、中立、客观。
"""


def encode_image(image_path):
    """将图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_a_dir = os.path.join(dataset_root, "A")
image_b_dir = os.path.join(dataset_root, "B")


# 定义一个提取数字的函数
def natural_key(string_):
    """将字符串中的数字部分转为整数，以便进行自然排序"""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string_)]


# 获取文件名并排序，确保 A 和 B 对应
# 使用 key 参数进行排序
filenames = sorted(
    [f for f in os.listdir(image_a_dir) if f.endswith(('.png', '.jpg', '.jpeg'))],
    key=natural_key
)

pbar = tqdm(filenames)
# 使用 'w' 模式打开文件，准备逐行写入
with open(output_file, "w", encoding="utf-8") as f_out:
    for fname in pbar:
        pbar.set_description(f"正在处理: {fname}")
        path_a = os.path.join(image_a_dir, fname)
        path_b = os.path.join(image_b_dir, fname)

        # 编码图片
        base64_a = encode_image(path_a)
        base64_b = encode_image(path_b)

        try:
            # 调用 vLLM 的 OpenAI 兼容接口
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            # Data URL 模式，它允许你直接把文件内容“写”在地址栏里
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_a}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_b}"}},
                        ],
                    }
                ],
                max_tokens=2048,
                temperature=0.7
            )

            answer = response.choices[0].message.content
            print(answer)

            # 构造 LLaMA-Factory 要求的 ShareGPT 格式
            # 注意：这里需要放图片的相对路径或绝对路径，训练时框架会去读
            shareGPT = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>\n<image>\n{prompt_text}"
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ],
                "images": [
                    os.path.abspath(path_a),  # 使用绝对路径最稳妥
                    os.path.abspath(path_b)
                ]
            }

            # 3. 将对象转为字符串并立即写入一行
            f_out.write(json.dumps(shareGPT, ensure_ascii=False) + "\n")
            f_out.flush()  # 强制刷新缓冲区，确保实时写入硬盘

        except Exception as e:
            print(f"\n跳过文件 {fname}，错误原因: {e}")
