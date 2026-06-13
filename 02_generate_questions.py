#!/usr/bin/env python3
"""
Phase 2: 基于图像描述生成多样化问题

输入: outputs/01_descriptions.jsonl
输出: outputs/02_questions.jsonl
格式: {image_path, dataset, description, questions: [{question_type, question}]}
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from src.scanner import load_processed_records
from src.vlm_client import VLMClient
from src.prompts import QUESTION_GENERATION_PROMPT


def parse_questions_response(response: str) -> list:
    """解析模型返回的问题列表"""
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
                questions.append({
                    "question_type": q_type,
                    "question": q_content
                })

    return questions


def main():
    parser = argparse.ArgumentParser(description="Phase 2: 生成多样化问题")
    parser.add_argument("--input", type=str, default="outputs/01_descriptions.jsonl",
                        help="Phase1输出的描述文件")
    parser.add_argument("--output", type=str, default="outputs/02_questions.jsonl",
                        help="输出文件路径")
    parser.add_argument("--num-questions", type=int, default=8,
                        help="每张图生成的问题数量")
    parser.add_argument("--min-questions", type=int, default=5,
                        help="每张图最少生成的问题数量")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称")
    parser.add_argument("--base-url", type=str, default="http://10.129.107.145:8001/v1",
                        help="vLLM服务地址")
    parser.add_argument("--retries", type=int, default=3,
                        help="失败重试次数")

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        print("请先运行 Phase 1: 01_generate_desc.py")
        return

    # 加载描述
    descriptions = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                descriptions.append(record)
            except:
                pass

    print("="*60)
    print("Phase 2: 生成多样化问题")
    print("="*60)
    print(f"加载描述: {len(descriptions)} 条")

    # 断点续传
    processed_images = load_processed_records(output_file, "image_path")
    if processed_images:
        print(f"发现已处理 {len(processed_images)} 张图像，跳过...")
        descriptions = [d for d in descriptions if d["image_path"] not in processed_images]
        print(f"剩余待处理: {len(descriptions)} 张图像")

    if not descriptions:
        print("\n所有图像已处理完成！")
        return

    # 初始化VLM客户端
    vlm = VLMClient(base_url=args.base_url, model=args.model, retries=args.retries)

    # 主处理循环
    success_count = 0
    total_questions = 0

    with open(output_file, "a", encoding="utf-8") as f_out:
        pbar = tqdm(descriptions)
        for desc in pbar:
            image_path = Path(desc["image_path"])
            dataset = desc["dataset"]
            description = desc["description"]

            pbar.set_description(f"处理中: {dataset}")

            try:
                prompt = QUESTION_GENERATION_PROMPT.format(
                    num_questions=args.num_questions,
                    min_questions=args.min_questions,
                    description=description
                )

                response = vlm.call(
                    prompt=prompt,
                    image_path=image_path,
                    max_tokens=768,
                    temperature=0.4
                )

                questions = parse_questions_response(response)

                if questions:
                    record = {
                        "image_path": desc["image_path"],
                        "dataset": dataset,
                        "description": description,
                        "questions": questions,
                        "generated_at": datetime.now().isoformat()
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()

                    success_count += 1
                    total_questions += len(questions)

                pbar.set_postfix({
                    "图像": success_count,
                    "总问题": total_questions,
                    "平均": f"{total_questions/success_count:.1f}" if success_count else 0
                })

            except Exception as e:
                print(f"\n[错误] {image_path.name}: {e}")
                continue

    print("\n" + "="*60)
    print("Phase 2 完成！")
    print(f"输出文件: {output_file}")
    print(f"处理图像: {success_count} 张")
    print(f"生成问题: {total_questions} 个")
    if success_count:
        print(f"平均每图: {total_questions/success_count:.1f} 个问题")
    print("="*60)


if __name__ == "__main__":
    main()
