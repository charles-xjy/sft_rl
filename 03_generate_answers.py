#!/usr/bin/env python3
"""
Phase 3: generate answers with analysis and save SFT-format output.

Input: outputs/02_questions.jsonl
Output: outputs/03_answers_sft.jsonl
"""
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.prompts import ANSWER_DIRECT_PROMPT, ANSWER_GENERATION_PROMPT
from src.validator import validate_answer
from src.vlm_client import VLMClient


def main():
    parser = argparse.ArgumentParser(description="Phase 3: 生成带分析过程的答案")
    parser.add_argument("--input", type=str, default="outputs/02_questions.jsonl", help="Phase 2 输出文件")
    parser.add_argument("--output", type=str, default="outputs/03_answers_sft.jsonl", help="输出文件路径")
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--base-url", type=str, default="http://10.129.107.145:8001/v1", help="vLLM 服务地址")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    parser.add_argument("--direct-ratio", type=float, default=0.3,
                        help="不带 analysis 的直答样本占比(打散固定起手式,缓解过度模板化)")
    parser.add_argument("--seed", type=int, default=42, help="直答/分析模式分配的随机种子")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    def pick_mode(qtype: str) -> str:
        """决定本条用 analysis 还是 direct 模式。
        unanswerable / spatial_relation 需要证据链,始终走 analysis;
        其余按 direct_ratio 随机直答。"""
        if qtype in ("unanswerable", "ambiguous", "spatial_relation"):
            return "analysis"
        return "direct" if rng.random() < args.direct_ratio else "analysis"

    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"错误: 输入文件不存在 {input_file}")
        print("请先运行 Phase 2: 02_generate_questions.py")
        return

    all_questions = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                for q in record["questions"]:
                    all_questions.append({
                        "image_path": record["image_path"],
                        "dataset": record["dataset"],
                        "description": record["description"],
                        "question_type": q["question_type"],
                        "question": q["question"],
                    })
            except Exception:
                pass

    print("=" * 60)
    print("Phase 3: 生成带分析过程的答案")
    print("=" * 60)
    print(f"加载问题: {len(all_questions)} 个")

    processed_keys = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    user_msg = record["messages"][0]["content"]
                    question = user_msg.replace("<image>", "").strip()
                    key = f"{record['images'][0]}|{question}"
                    processed_keys.add(key)
                except Exception:
                    pass

    if processed_keys:
        print(f"发现已处理 {len(processed_keys)} 个问答，跳过")
        all_questions = [
            q for q in all_questions if f"{q['image_path']}|{q['question']}" not in processed_keys
        ]
        print(f"剩余待处理 {len(all_questions)} 个问答")

    if not all_questions:
        print("\n所有问答已处理完成")
        return

    vlm = VLMClient(base_url=args.base_url, model=args.model, retries=args.retries)

    stats = defaultdict(int)
    success_count = 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        pbar = tqdm(all_questions)
        for q_item in pbar:
            image_path = Path(q_item["image_path"])
            dataset = q_item["dataset"]
            description = q_item["description"]
            question_type = q_item["question_type"]
            question = q_item["question"]
            pbar.set_description(f"处理中 {dataset} | {question_type}")

            try:
                mode = pick_mode(question_type)
                template = ANSWER_GENERATION_PROMPT if mode == "analysis" else ANSWER_DIRECT_PROMPT
                prompt = template.format(
                    question=question,
                    question_type=question_type,
                    description=description,
                )
                answer = vlm.call(
                    prompt=prompt,
                    image_path=image_path,
                    max_tokens=4096,
                    temperature=0.3,
                )

                if not answer:
                    print(f"\n[empty-output] {image_path.name}: empty answer for question: {question[:80]}")
                    continue

                is_valid, warnings, auto_label = validate_answer(
                    answer, question_type, expect_analysis=(mode == "analysis")
                )
                record = {
                    "messages": [
                        {"role": "user", "content": f"<image>\n{question}"},
                        {"role": "assistant", "content": answer},
                    ],
                    "images": [str(image_path.resolve())],
                    "meta": {
                        "task_type": "vqa_with_analysis" if mode == "analysis" else "vqa_direct",
                        "answer_mode": mode,
                        "question_type": question_type,
                        "source_dataset": dataset,
                        "auto_validation": {
                            "is_valid": is_valid,
                            "warnings": warnings,
                            "label": auto_label,
                        },
                        "generated_at": datetime.now().isoformat(),
                    },
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                success_count += 1
                stats["total"] += 1
                stats[f"dataset_{dataset}"] += 1
                stats[f"type_{question_type}"] += 1
                stats[f"mode_{mode}"] += 1
                if auto_label == "pass":
                    stats["auto_pass"] += 1

                pbar.set_postfix({"generated": success_count, "auto_pass": stats["auto_pass"]})
            except Exception as e:
                print(f"\n[error] {question[:30]}...: {e}")
                continue

    print("\n" + "=" * 60)
    print("Phase 3 完成")
    print(f"输出文件: {output_file}")
    print(f"总样本数: {stats['total']}")
    pass_rate = stats["auto_pass"] / max(stats["total"], 1) * 100
    print(f"自动质检通过: {stats['auto_pass']} ({pass_rate:.1f}%)")
    print("\n按数据集分布:")
    for key, count in stats.items():
        if key.startswith("dataset_"):
            print(f"  {key[8:]}: {count}")
    print("\n按问题类型分布:")
    for key, count in stats.items():
        if key.startswith("type_"):
            print(f"  {key[5:]}: {count}")
    print("=" * 60)

    stats_file = output_file.parent / "03_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, ensure_ascii=False, indent=2)
    print(f"统计信息已保存: {stats_file}")


if __name__ == "__main__":
    main()
