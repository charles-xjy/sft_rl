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
dataset_root = "dataset/SECOND/train"
# 获取当前时间并格式化为: 月.日.时:分

current_time = datetime.now().strftime("%m.%d.%H:%M")
output_file = f"{current_time}_SECOND_sft.jsonl"

prompt_text = """
    # Role
    你是一位精通遥感影像分析与城市规划的资深专家。你的任务是对比提供的两张时序卫星影像（图 A 为前期，图 B 为后期），精准识别地物变化，判定开发模式，并从城市治理角度提供专业评估与政策建议。
    
    # Task Steps
    
    ### 第一步：客观变化描述 (Objective Description)
    请从视觉特征出发，描述图 B 相较于图 A 的变化：
    1. **地物类型精细识别**：基于光谱、纹理及阴影特征识别新增地物。
       - *要求*：必须引用视觉证据。例如：“白色高亮度矩形（判定为工业厂房）”、“红褐色规则阵列点状物（判定为住宅区）”、“深灰色线性连续纹理（判定为硬化道路）”。
    2. **几何与空间分布特征**：
       - 描述变化的形态：是聚落状、线性扩展还是点状散布？
       - 描述空间关系：变化区域在图中的方位、建筑密度（高/中/低）、是否沿原有交通干线分布、是否有明显的红线边界。
    
    ### 第二步：开发模式识别 (Pattern Recognition)
    基于第一步的几何证据，将开发性质判定为以下两者之一，并说明理由。禁止在缺乏证据的情况下直接假设：
    - **模式 A：有序规划型**。判定证据需包含：路网呈网格状或环状且与建筑同步、建筑退界规整、具备配套绿地或公共广场。反映了【存量土地优化】或【高水平城镇化】。
    - **模式 B：自发蔓延型**。判定证据需包含：建筑零散分布于林地/耕地边缘（飞地扩张）、缺乏系统性路网连接、原有生态廊道或连续植被斑块被切断。反映了【无序扩张】或【监管盲区】。
    
    ### 第三步：多维治理评估 (Governance Assessment)
    结合影像中的空间变化，从对立维度进行客观评价：
    1. **正面评价（土地价值与民生）**：分析该开发如何提升了土地利用率（Land Use Efficiency）、改善了区域居住水平或通过产业导入促进了局部经济。
    2. **潜在风险（生态与负荷）**：识别视觉上可感知的风险。例如：大面积硬化地面可能加剧热岛效应；建筑侵占河道或绿色斑块导致的生态连通性下降；新增人口对周边基础设施造成的潜在压力。
    
    ### 第四步：精准政策建议 (Policy Recommendations)
    建议必须与第二步的判定结论形成强逻辑对应：
    - **针对“有序开发”**：建议侧重于“精细化运营”。如：建设智慧社区管理平台、实施能源动态监测、优化交通信号微循环。
    - **针对“无序开发”**：建议侧重于“规划执法与修复”。如：立即启动遥感常态化违法建设监测、执行“占补平衡”生态修复、划定生态红线刚性约束。
    
    # Constraints
    1. **专业性**：使用遥感与城市规划术语（如：光谱特征、容积率、生态斑块连通性）。
    2. **客观性**：严禁过度推测。若影像模糊，请使用“疑似”、“推测可能”等词汇。
    3. **简洁性**：直接输出分析内容，不要带有任何开场白或解释性文字。
    4. **针对性**：所有分析需要结合图像内容，输出内容不要过于宽泛、泛化，缺乏针对性
"""


def encode_image(image_path):
    """将图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_a_dir = os.path.join(dataset_root, "im1")
image_b_dir = os.path.join(dataset_root, "im2")


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
        system_text="""
        你现在是一名城市治理的专家。这是城市发展对比图。请先识别新增地物的分布特征,并据此给出城市治理建议
        """.strip()
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
            # print(answer)

            # 构造 LLaMA-Factory 要求的 ShareGPT 格式
            # 注意：这里需要放图片的相对路径或绝对路径，训练时框架会去读
            shareGPT = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>\n<image>\n{system_text}"
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
