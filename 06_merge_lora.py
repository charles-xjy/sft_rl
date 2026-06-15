#!/usr/bin/env python3
"""
Merge a trained Unsloth Qwen3-VL LoRA adapter into a full 16-bit model.

Typical usage:
    python 06_merge_lora.py \
      --adapter outputs/qwen3vl-sft-lora \
      --out outputs/qwen3vl-sft-lora_merged

You can also merge a specific checkpoint directory:
    python 06_merge_lora.py \
      --adapter outputs/qwen3vl-sft-lora/checkpoint-364 \
      --out outputs/qwen3vl-sft-lora-ckpt364-merged
"""

import argparse
from pathlib import Path

from unsloth import FastVisionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a Qwen3-VL LoRA adapter into a full 16-bit model."
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="outputs/qwen3vl-sft-lora",
        help="Path to the LoRA adapter directory or a checkpoint directory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for the merged model. Defaults to <adapter>_merged.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the adapter in 4-bit before merging. Leave off for normal 16-bit LoRA training outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    out_dir = Path(args.out) if args.out else Path(str(adapter_dir) + "_merged")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Merge Qwen3-VL LoRA")
    print("=" * 64)
    print(f"adapter : {adapter_dir}")
    print(f"output  : {out_dir}")
    print(f"4bit    : {args.load_in_4bit}")
    print("=" * 64)

    model, tokenizer = FastVisionModel.from_pretrained(
        str(adapter_dir),
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
    )

    model.save_pretrained_merged(
        str(out_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"[saved] merged model -> {out_dir}")


if __name__ == "__main__":
    main()
