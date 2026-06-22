#!/usr/bin/env python3
"""Preview and convert the first N GeoChat_Instruct samples to JSONL."""

import argparse
import json
from pathlib import Path


def extract_pair(item: dict) -> tuple[str, str]:
    conversations = item.get("conversations") or []
    if len(conversations) < 2:
        raise ValueError("missing conversations")

    question = conversations[0].get("value", "").replace("<image>", "").strip()
    reference = conversations[1].get("value", "").strip()
    return question, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="dataset/GeoChat_Instruct/GeoChat_Instruct.json",
        help="Path to GeoChat_Instruct.json",
    )
    parser.add_argument(
        "--image-root",
        default="dataset/GeoChat_Instruct",
        help="Directory that contains the extracted GeoChat images",
    )
    parser.add_argument(
        "--output",
        default="scratch/geochat_first100.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to convert")
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Record whether each converted image path exists",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    image_root = Path(args.image_root).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in data[: args.limit]:
            try:
                question, reference = extract_pair(item)
            except ValueError:
                skipped += 1
                continue

            image_rel = item.get("image") or item.get("id")
            if not image_rel:
                skipped += 1
                continue

            image_path = image_root / image_rel
            row = {
                "id": item.get("id"),
                "image": str(image_path),
                "question": question,
                "reference": reference,
                "source": "GeoChat_Instruct",
            }
            if args.check_images:
                row["image_exists"] = image_path.exists()

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"input: {input_path}")
    print(f"total records in source: {len(data)}")
    print(f"written: {written}")
    print(f"skipped: {skipped}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
