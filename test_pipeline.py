#!/usr/bin/env python3
"""
Test the full single-image pipeline with the same logic as the main generators.

Usage:
    python test_pipeline.py [image_path]
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from src.prompts import (
    ANSWER_DIRECT_PROMPT,
    ANSWER_GENERATION_PROMPT,
    DESCRIPTION_PROMPT,
    QUESTION_GENERATION_PROMPT,
)
from src.sampling import assign_question_types, format_plan
from src.validator import validate_answer
from src.vlm_client import VLMClient


VALID_QUESTION_TYPES = {
    "existence",
    "attribute",
    "location",
    "spatial_relation",
    "counting",
    "unanswerable",
    "ambiguous",
}


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


def parse_questions_response(response: str) -> list[dict[str, str]]:
    questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        q_type, q_content = line.split("|", 1)
        q_type = q_type.strip().lower()
        q_content = q_content.strip()
        if q_type in VALID_QUESTION_TYPES and q_content:
            questions.append({"question_type": q_type, "question": q_content})
    return questions


def pick_answer_mode(
    image_path: str,
    question: str,
    question_type: str,
    direct_ratio: float,
    seed: int,
) -> str:
    if question_type in ("unanswerable", "ambiguous", "spatial_relation"):
        return "analysis"
    digest = hashlib.md5(f"{seed}|{image_path}|{question}".encode("utf-8")).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return "direct" if frac < direct_ratio else "analysis"


print("=" * 80)
print("测试单图三阶段流水线")
print("=" * 80)

parser = argparse.ArgumentParser(description="测试单图三阶段流水线")
parser.add_argument("image_path", nargs="?", default=None, help="测试图片路径")
parser.add_argument("--dataset-root", default="dataset", help="数据集根目录")
parser.add_argument("--min-questions", type=int, default=2, help="最少问题数")
parser.add_argument("--num-questions", type=int, default=5, help="最多问题数")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--direct-ratio", type=float, default=0.3, help="直答样本占比")
parser.add_argument("--model", type=str, default=None, help="模型名称")
parser.add_argument("--base-url", type=str, default="http://10.129.107.145:8001/v1", help="vLLM 服务地址")
parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
args = parser.parse_args()

print("\n[1/6] 查找测试图片...")
test_image = Path(args.image_path) if args.image_path else find_test_image(Path(args.dataset_root))
if not test_image or not test_image.exists():
    print("  未找到测试图片，请手动指定: python test_pipeline.py /path/to/image.png")
    sys.exit(1)
test_image = test_image.resolve()
print(f"  测试图片: {test_image}")
print(f"  大小: {test_image.stat().st_size / 1024:.1f} KB")

print("\n[2/6] 初始化 VLM 客户端...")
try:
    vlm = VLMClient(base_url=args.base_url, model=args.model, retries=args.retries)
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
        max_tokens=2048,
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
    assigned = assign_question_types(
        str(test_image), args.min_questions, args.num_questions, args.seed
    )
    print("  建议题型配额:")
    print(format_plan(assigned))
    question_prompt = QUESTION_GENERATION_PROMPT.format(
        assigned_plan=format_plan(assigned),
        description=description,
    )
    response = vlm.call(
        prompt=question_prompt,
        image_path=test_image,
        max_tokens=2048,
        temperature=0.4,
        dump_response=True,
    )
    questions = parse_questions_response(response)
    if len(questions) > args.num_questions:
        questions = questions[: args.num_questions]
    print("-" * 80)
    for i, q in enumerate(questions, 1):
        print(f"{i}. [{q['question_type']}] {q['question']}")
    print("-" * 80)
    if not questions:
        print("  未解析到有效问题")
        sys.exit(1)
except Exception as e:
    print(f"  问题生成失败: {e}")
    sys.exit(1)

print("\n[5/6] Phase 3: 生成答案...")
records = []
for i, q in enumerate(questions, 1):
    mode = pick_answer_mode(
        str(test_image), q["question"], q["question_type"], args.direct_ratio, args.seed
    )
    template = ANSWER_GENERATION_PROMPT if mode == "analysis" else ANSWER_DIRECT_PROMPT
    print(f"\n  Question {i}: [{q['question_type']}] {q['question']}")
    print(f"  answer_mode: {mode}")
    try:
        answer_prompt = template.format(
            question=q["question"],
            question_type=q["question_type"],
            description=description,
        )
        answer = vlm.call(
            prompt=answer_prompt,
            image_path=test_image,
            max_tokens=2048,
            temperature=0.3,
            dump_response=True,
        )
        print("-" * 80)
        print(answer)
        print("-" * 80)

        is_valid, warnings, auto_label = validate_answer(
            answer, q["question_type"], expect_analysis=(mode == "analysis")
        )
        print(f"  自动质检: {'通过' if is_valid else '不通过'}")
        print(f"  标签: {auto_label}")
        if warnings:
            for warning in warnings:
                print(f"    - {warning}")

        records.append(
            {
                "messages": [
                    {"role": "user", "content": f"<image>\n{q['question']}"},
                    {"role": "assistant", "content": answer},
                ],
                "images": [str(test_image)],
                "meta": {
                    "task_type": "vqa_with_analysis" if mode == "analysis" else "vqa_direct",
                    "answer_mode": mode,
                    "question_type": q["question_type"],
                    "auto_validation": {
                        "is_valid": is_valid,
                        "warnings": warnings,
                        "label": auto_label,
                    },
                    "generated_at": datetime.now().isoformat(),
                },
            }
        )
    except Exception as e:
        print(f"  答案生成失败: {e}")
        sys.exit(1)

print("\n[6/6] SFT 预览...")
preview = {
    "image": str(test_image),
    "num_questions": len(questions),
    "records": records[:2],
}
preview_json = json.dumps(preview, ensure_ascii=False, indent=2)
print(preview_json[:1600] + ("..." if len(preview_json) > 1600 else ""))
print("=" * 80)
