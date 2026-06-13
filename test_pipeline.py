#!/usr/bin/env python3
"""
测试三阶段流水线：单张图片走完全流程

用法:
    python test_pipeline.py [图片路径]
    python test_pipeline.py --dataset dataset/LEVIR-CD+/train/B/xxx.png
"""
import argparse
import json
import sys
from pathlib import Path

print("="*80)
print("测试三阶段VQA数据构建流水线")
print("="*80)

# 1. 导入模块
print("\n[1/6] 导入模块...")
try:
    from src.vlm_client import VLMClient, detect_model
    from src.prompts import DESCRIPTION_PROMPT, QUESTION_GENERATION_PROMPT, ANSWER_GENERATION_PROMPT
    from src.validator import validate_answer
    print("  ✓ 模块导入成功")
except Exception as e:
    print(f"  ✗ 模块导入失败: {e}")
    sys.exit(1)

# 2. 解析参数
parser = argparse.ArgumentParser(description="测试三阶段流水线")
parser.add_argument("image_path", nargs="?", help="测试图片路径", default=None)
parser.add_argument("--dataset-root", default="dataset", help="数据集根目录")
args = parser.parse_args()

# 3. 查找测试图片
print("\n[2/6] 查找测试图片...")
if args.image_path:
    test_image = Path(args.image_path)
else:
    # 自动找第一张图片
    dataset_root = Path(args.dataset_root)
    patterns = [
        "EBD/**/*_post_disaster.*",
        "LEVIR-CD+/**/B/*.png",
        "SECOND/**/im2/*.png",
    ]
    test_image = None
    for pattern in patterns:
        for ext in ["png", "jpg", "jpeg"]:
            p = pattern.replace("*.", f"*.{ext}")
            images = list(dataset_root.glob(p))
            if images:
                test_image = images[0]
                break
        if test_image:
            break

if not test_image or not test_image.exists():
    print(f"  ✗ 未找到测试图片，请手动指定: python test_pipeline.py /path/to/image.png")
    sys.exit(1)

print(f"  ✓ 测试图片: {test_image}")
print(f"    大小: {test_image.stat().st_size / 1024:.1f} KB")

# 4. 初始化VLM客户端
print("\n[3/6] 初始化VLM客户端...")
try:
    vlm = VLMClient()
    print(f"  ✓ 服务地址: {vlm.base_url}")
    print(f"  ✓ 使用模型: {vlm.model}")
except Exception as e:
    print(f"  ✗ VLM客户端初始化失败: {e}")
    print("\n请检查:")
    print("  1. vLLM服务是否启动")
    print("  2. 网络是否通畅: curl http://10.129.107.145:8001/v1/models")
    sys.exit(1)

# 5. Phase 1: 生成描述
print("\n[4/6] Phase 1: 生成图像描述...")
try:
    description = vlm.call(
        prompt=DESCRIPTION_PROMPT,
        image_path=test_image,
        max_tokens=512,
        temperature=0.3
    )
    print("\n" + "-"*80)
    print("📝 图像描述:")
    print("-"*80)

    if description is None or description.strip() == "":
        print("  ⚠️  模型返回了空描述！")
        print("  ❓ 可能原因:")
        print("      - 部署的模型不是多模态模型（缺少 -VL 后缀）")
        print("      - 当前模型: " + vlm.model)
        print("      - 请部署 Qwen/Qwen3.5-VL-27B-Instruct")
        print("-"*80)
        sys.exit(1)

    print(description)
    print("-"*80)
except Exception as e:
    print(f"  ✗ 描述生成失败: {e}")
    sys.exit(1)

# 6. Phase 2: 生成问题
print("\n[5/6] Phase 2: 生成多样化问题...")
try:
    prompt = QUESTION_GENERATION_PROMPT.format(
        num_questions=5,
        min_questions=3,
        description=description
    )
    response = vlm.call(
        prompt=prompt,
        image_path=test_image,
        max_tokens=768,
        temperature=0.4
    )

    # 解析问题
    questions = []
    valid_types = {"existence", "attribute", "location", "spatial_relation",
                   "counting", "scene", "comparison", "reasoning"}

    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 1)
        if len(parts) == 2:
            q_type, q_content = parts
            q_type = q_type.strip().lower()
            q_content = q_content.strip()
            if q_type in valid_types and q_content:
                questions.append({"type": q_type, "content": q_content})

    print("\n" + "-"*80)
    print(f"❓ 生成的问题 ({len(questions)} 个):")
    print("-"*80)
    for i, q in enumerate(questions, 1):
        print(f"  {i}. [{q['type']:15s}] {q['content']}")
    print("-"*80)

    if not questions:
        print("  ✗ 未生成有效问题")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ 问题生成失败: {e}")
    sys.exit(1)

# 7. Phase 3: 生成答案（选第一个问题测试）
print("\n[6/6] Phase 3: 生成带分析的答案...")
test_question = questions[0]
print(f"  测试问题: [{test_question['type']}] {test_question['content']}")

try:
    prompt = ANSWER_GENERATION_PROMPT.format(
        question=test_question['content'],
        question_type=test_question['type'],
        description=description
    )
    answer = vlm.call(
        prompt=prompt,
        image_path=test_image,
        max_tokens=768,
        temperature=0.3
    )

    print("\n" + "-"*80)
    print("💡 生成的答案:")
    print("-"*80)
    print(answer)
    print("-"*80)

    # 自动质检
    is_valid, warnings, auto_label = validate_answer(answer, test_question['type'])
    print(f"\n🔍 自动质检结果:")
    print(f"  状态: {'✓ 通过' if is_valid else '✗ 不通过'}")
    print(f"  标签: {auto_label}")
    if warnings:
        print(f"  警告:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"  警告: 无")

except Exception as e:
    print(f"  ✗ 答案生成失败: {e}")
    sys.exit(1)

# 8. 总结
print("\n" + "="*80)
print("✅ 流水线测试完成！")
print("="*80)
print(f"""
测试总结:
  - 图像路径: {test_image}
  - 描述生成长度: {len(description)} 字符
  - 生成问题数: {len(questions)} 个
  - 问题类型分布: {', '.join(set(q['type'] for q in questions))}
  - 答案自动质检: {'通过' if is_valid else '不通过'}
""")

# 显示最终SFT格式示例
print("📋 最终SFT格式预览:")
sft_sample = {
    "messages": [
        {
            "role": "user",
            "content": f"<image>\n{test_question['content']}"
        },
        {
            "role": "assistant",
            "content": answer
        }
    ],
    "images": [str(test_image.resolve())],
    "meta": {
        "task_type": "vqa_with_analysis",
        "question_type": test_question['type'],
        "auto_validation": {
            "is_valid": is_valid,
            "warnings": warnings,
            "label": auto_label
        }
    }
}
print(json.dumps(sft_sample, ensure_ascii=False, indent=2)[:500] + "...")
print("="*80)
