#!/usr/bin/env python3
"""Extract GeoChat_Instruct samples that are easy to score for GRPO."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


EASY_TYPES = {
    "scene_classification",
    "object_presence_yesno",
    "comparison_yesno",
    "count_comparison_yesno",
    "flood_condition",
    "rural_urban_classification",
}


def question_of(item: dict) -> str:
    conversations = item.get("conversations") or []
    if not conversations:
        return ""
    return conversations[0].get("value", "").replace("<image>", "").strip()


def answer_of(item: dict) -> str:
    conversations = item.get("conversations") or []
    if len(conversations) < 2:
        return ""
    return conversations[1].get("value", "").strip()


def classify_question(question: str) -> str | None:
    q = " ".join(question.split())
    ql = q.lower()

    if ql.startswith("classify the given image") or "one of the following classes" in ql:
        return "scene_classification"

    if re.match(r"is it a rural or an urban area", ql):
        return "rural_urban_classification"

    if re.search(r"\bflooded\b|\bnon flooded\b|\bflood\b", ql):
        return "flood_condition"

    if re.match(r"is|are|does|do|can", ql):
        if re.search(r"\bnumber of\b|\bamount of\b", ql) and re.search(
            r"\bequal\b|\bmore\b|\bless\b|\bthan\b", ql
        ):
            return "count_comparison_yesno"
        if re.search(r"\bmore\b|\bless\b|\blarger\b|\bsmaller\b|\bthan\b", ql):
            return "comparison_yesno"
        if not re.search(
            r"\bleft\b|\bright\b|\btop\b|\bbottom\b|\bnear\b|\baround\b|\badjacent\b|\bbetween\b|\binside\b|\bsurround\b|\blocated\b|\bat the\b",
            ql,
        ):
            return "object_presence_yesno"

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/GeoChat_Instruct/GeoChat_Instruct.json")
    parser.add_argument("--image-root", default="dataset/GeoChat_Instruct")
    parser.add_argument("--output", default="scratch/geochat_easy_grpo_1000_per_type.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--per-type",
        type=int,
        default=1000,
        help="Keep up to N samples for each easy question type. 0 disables per-type sampling.",
    )
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    image_root = Path(args.image_root).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    counts = Counter()
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            question = question_of(item)
            question_type = classify_question(question)
            if question_type not in EASY_TYPES:
                continue
            if args.per_type and counts[question_type] >= args.per_type:
                if all(counts[t] >= args.per_type for t in EASY_TYPES):
                    break
                continue

            image_rel = item.get("image") or item.get("id")
            reference = answer_of(item)
            if not image_rel or not question or not reference:
                continue

            image_path = image_root / image_rel
            row = {
                "id": item.get("id"),
                "image": str(image_path),
                "question": question,
                "reference": reference,
                "question_type": question_type,
                "source": "GeoChat_Instruct",
            }
            if args.check_images:
                row["image_exists"] = image_path.exists()

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[question_type] += 1
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"written: {written}")
    for key, value in counts.most_common():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
