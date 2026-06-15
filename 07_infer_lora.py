#!/usr/bin/env python3
"""
Run local single-image inference with a Qwen3-VL LoRA adapter without merging.

Typical usage:
    python 07_infer_lora.py \
      --adapter outputs/qwen3vl-sft-lora \
      --image /path/to/example.png \
      --question "图中右侧道路附近是否有建筑？"
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from unsloth import FastVisionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local inference with a Qwen3-VL LoRA adapter without merging."
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="outputs/qwen3vl-sft-lora",
        help="Path to the LoRA adapter directory or a checkpoint directory.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for deterministic decoding.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.1,
        help="min_p sampling parameter used when temperature > 0.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the adapter in 4-bit. Only use this if the training/export setup requires it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter)
    image_path = Path(args.image)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")

    print("=" * 64)
    print("Load Qwen3-VL LoRA For Inference")
    print("=" * 64)
    print(f"adapter : {adapter_dir}")
    print(f"image   : {image_path}")
    print("=" * 64)

    model, tokenizer = FastVisionModel.from_pretrained(
        str(adapter_dir),
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.question},
                {"type": "image"},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        image,
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    if args.temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "min_p": args.min_p,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    answer_ids = output[0][prompt_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    print("\n[question]")
    print(args.question)
    print("\n[answer]")
    print(answer)


if __name__ == "__main__":
    main()
