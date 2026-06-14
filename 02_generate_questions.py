#!/usr/bin/env python3
"""
Phase 2: generate diverse questions from image descriptions.

Input: outputs/01_descriptions.jsonl
Output: outputs/02_questions.jsonl
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.prompts import QUESTION_GENERATION_PROMPT
from src.sampling import assign_question_types, format_plan
from src.scanner import load_processed_records
from src.vlm_client import VLMClient


def parse_questions_response(response: str) -> list:
    """Parse `type|question` lines from model output."""
    questions = []
    valid_types = {
        "existence",
        "attribute",
        "location",
        "spatial_relation",
        "counting",
        "unanswerable",
        "ambiguous",
    }

    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        q_type, q_content = line.split("|", 1)
        q_type = q_type.strip().lower()
        q_content = q_content.strip()
        if q_type in valid_types and q_content:
            questions.append({"question_type": q_type, "question": q_content})

    return questions


def main():
    parser = argparse.ArgumentParser(description="Phase 2: 生成多样化问题")
    parser.add_argument("--input", type=str, default="outputs/01_descriptions.jsonl", help="Phase 1 输出文件")
    parser.add_argument("--output", type=str, default="outputs/02_questions.jsonl", help="输出文件路径")
    parser.add_argument("--num-questions", type=int, default=5, help="每张图像最多生成的问题数量")
    parser.add_argument("--min-questions", type=int, default=2, help="每张图像最少生成的问题数量")
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--base-url", type=str, default="http://10.129.107.145:8001/v1", help="vLLM 服务地址")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    parser.add_argument("--seed", type=int, default=42, help="问题类型软配额分配的随机种子")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"错误: 输入文件不存在 {input_file}")
        print("请先运行 Phase 1: 01_generate_desc.py")
        return

    descriptions = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                descriptions.append(json.loads(line.strip()))
            except Exception:
                pass

    print("=" * 60)
    print("Phase 2: 生成多样化问题")
    print("=" * 60)
    print(f"加载描述: {len(descriptions)} 条")

    processed_images = load_processed_records(output_file, "image_path")
    if processed_images:
        print(f"发现已处理 {len(processed_images)} 张图像，跳过")
        descriptions = [d for d in descriptions if d["image_path"] not in processed_images]
        print(f"剩余待处理 {len(descriptions)} 张图像")

    if not descriptions:
        print("\n所有图像已处理完成")
        return

    vlm = VLMClient(base_url=args.base_url, model=args.model, retries=args.retries)

    success_count = 0
    total_questions = 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        pbar = tqdm(descriptions)
        for desc in pbar:
            image_path = Path(desc["image_path"])
            dataset = desc["dataset"]
            description = desc["description"]
            pbar.set_description(f"处理中 {dataset}")

            try:
                assigned = assign_question_types(
                    desc["image_path"], args.min_questions, args.num_questions, args.seed
                )
                prompt = QUESTION_GENERATION_PROMPT.format(
                    assigned_plan=format_plan(assigned),
                    description=description,
                )
                response = vlm.call(
                    prompt=prompt,
                    image_path=image_path,
                    max_tokens=4096,
                    temperature=0.4,
                )

                questions = parse_questions_response(response)
                if len(questions) > args.num_questions:
                    questions = questions[: args.num_questions]
                if questions:
                    record = {
                        "image_path": desc["image_path"],
                        "dataset": dataset,
                        "description": description,
                        "questions": questions,
                        "generated_at": datetime.now().isoformat(),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    success_count += 1
                    total_questions += len(questions)
                else:
                    print(f"\n[empty-output] {image_path.name}: no valid questions parsed from model response")

                avg_questions = f"{total_questions / success_count:.1f}" if success_count else "0"
                pbar.set_postfix({"images": success_count, "questions": total_questions, "avg": avg_questions})
            except Exception as e:
                print(f"\n[error] {image_path.name}: {e}")
                continue

    print("\n" + "=" * 60)
    print("Phase 2 完成")
    print(f"输出文件: {output_file}")
    print(f"处理图像: {success_count} 张")
    print(f"生成问题: {total_questions} 个")
    if success_count:
        print(f"平均每图: {total_questions / success_count:.1f} 个问题")
    print("=" * 60)


if __name__ == "__main__":
    main()
