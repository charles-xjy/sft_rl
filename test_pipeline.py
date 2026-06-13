#!/usr/bin/env python3
"""
Test the full single-image pipeline and print raw model responses.

Usage:
    python test_pipeline.py [image_path]
"""
import argparse
import json
import sys
from pathlib import Path

from src.prompts import ANSWER_GENERATION_PROMPT, DESCRIPTION_PROMPT, QUESTION_GENERATION_PROMPT
from src.validator import validate_answer
from src.vlm_client import VLMClient


def find_test_image(dataset_root: Path) -> Path | None:
    patterns = [
        "EBD/**/*_post_disaster.*",
        "LEVIR-CD+/**/B/*.png",
        "SECOND/**/im2/*.png",
    ]
    for pattern in patterns:
        for ext in ["png", "jpg", "jpeg"]:
            candidate_pattern = pattern.replace("*.", f"*.{ext}")
            images = list(dataset_root.glob(candidate_pattern))
            if images:
                return images[0]
    return None


def parse_questions(response: str) -> list[dict]:
    questions = []
    valid_types = {
        "existence",
        "attribute",
        "location",
        "spatial_relation",
        "counting",
        "scene",
        "comparison",
        "reasoning",
    }
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        q_type, q_content = line.split("|", 1)
        q_type = q_type.strip().lower()
        q_content = q_content.strip()
        if q_type in valid_types and q_content:
            questions.append({"type": q_type, "content": q_content})
    return questions


print("=" * 80)
print("测试三阶段 VQA 流水线")
print("=" * 80)

parser = argparse.ArgumentParser(description="测试三阶段流水线")
parser.add_argument("image_path", nargs="?", default=None, help="测试图片路径")
parser.add_argument("--dataset-root", default="dataset", help="数据集根目录")
args = parser.parse_args()

print("\n[1/6] 查找测试图片...")
test_image = Path(args.image_path) if args.image_path else find_test_image(Path(args.dataset_root))
if not test_image or not test_image.exists():
    print("  未找到测试图片，请手动指定: python test_pipeline.py /path/to/image.png")
    sys.exit(1)
print(f"  测试图片: {test_image}")
print(f"  大小: {test_image.stat().st_size / 1024:.1f} KB")

print("\n[2/6] 初始化 VLM 客户端...")
try:
    vlm = VLMClient()
    print(f"  服务地址: {vlm.base_url}")
    print(f"  使用模型: {vlm.model}")
except Exception as e:
    print(f"  VLM 客户端初始化失败: {e}")
    sys.exit(1)

print("\n[3/6] Phase 1: 生成描述...")
try:
    description = vlm.call(
        prompt=DESCRIPTION_PROMPT,
        image_path=test_image,
        max_tokens=512,
        temperature=0.3,
        dump_response=True,
    )
    print("-" * 80)
    print(description)
    print("-" * 80)
except Exception as e:
    print(f"  描述生成失败: {e}")
    sys.exit(1)

print("\n[4/6] Phase 2: 生成问题...")
try:
    question_prompt = QUESTION_GENERATION_PROMPT.format(
        num_questions=5,
        min_questions=3,
        description=description,
    )
    response = vlm.call(
        prompt=question_prompt,
        image_path=test_image,
        max_tokens=768,
        temperature=0.4,
        dump_response=True,
    )
    questions = parse_questions(response)
    print("-" * 80)
    for i, q in enumerate(questions, 1):
        print(f"{i}. [{q['type']}] {q['content']}")
    print("-" * 80)
    if not questions:
        print("  未解析到有效问题")
        sys.exit(1)
except Exception as e:
    print(f"  问题生成失败: {e}")
    sys.exit(1)

print("\n[5/6] Phase 3: 生成答案...")
test_question = questions[0]
print(f"  测试问题: [{test_question['type']}] {test_question['content']}")
try:
    answer_prompt = ANSWER_GENERATION_PROMPT.format(
        question=test_question["content"],
        question_type=test_question["type"],
        description=description,
    )
    answer = vlm.call(
        prompt=answer_prompt,
        image_path=test_image,
        max_tokens=768,
        temperature=0.3,
        dump_response=True,
    )
    print("-" * 80)
    print(answer)
    print("-" * 80)

    is_valid, warnings, auto_label = validate_answer(answer, test_question["type"])
    print(f"自动质检: {'通过' if is_valid else '不通过'}")
    print(f"标签: {auto_label}")
    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
except Exception as e:
    print(f"  答案生成失败: {e}")
    sys.exit(1)

print("\n[6/6] SFT 预览...")
sft_sample = {
    "messages": [
        {"role": "user", "content": f"<image>\n{test_question['content']}"},
        {"role": "assistant", "content": answer},
    ],
    "images": [str(test_image.resolve())],
    "meta": {
        "task_type": "vqa_with_analysis",
        "question_type": test_question["type"],
        "auto_validation": {
            "is_valid": is_valid,
            "warnings": warnings,
            "label": auto_label,
        },
    },
}
print(json.dumps(sft_sample, ensure_ascii=False, indent=2)[:800] + "...")
print("=" * 80)
