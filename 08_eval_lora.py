#!/usr/bin/env python3
"""
用训练好的 LoRA 适配器跑验证集，输出 val_pred3.jsonl。

每行格式与 val_pred2.jsonl 一致：
  {image, question, question_type, reference, prediction}
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from unsloth import FastVisionModel


def parse_args():
    p = argparse.ArgumentParser(description="LoRA 验证集批量推理")
    p.add_argument("--adapter", type=str, default="outputs/qwen3vl-sft-lora")
    p.add_argument("--val-file", type=str, default="outputs/sft/val.jsonl")
    p.add_argument("--output", type=str, default="outputs/baseline/val_pred3.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--load-in-4bit", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("Loading model + LoRA adapter...")
    model, tokenizer = FastVisionModel.from_pretrained(
        str(adapter_dir),
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    print(f"Adapter: {adapter_dir}")

    # 加载验证集
    val_dataset = load_dataset("json", data_files=args.val_file, split="train")
    print(f"Val samples: {len(val_dataset)}")

    # 逐条推理
    results = []
    for sample in tqdm(val_dataset, desc="Inference"):
        image_path = sample["image"]
        question = sample["question"]
        question_type = sample["question_type"]
        reference = sample["answer"]

        # 构建 messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        image = Image.open(image_path).convert("RGB")
        inputs = tokenizer(image, prompt, add_special_tokens=False, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generate_kwargs = {"max_new_tokens": args.max_new_tokens, "use_cache": True}
        if args.temperature > 0:
            generate_kwargs.update(do_sample=True, temperature=args.temperature, min_p=0.1)
        else:
            generate_kwargs["do_sample"] = False

        with torch.inference_mode():
            output = model.generate(**inputs, **generate_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        prediction = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

        results.append({
            "image": image_path,
            "question": question,
            "question_type": question_type,
            "reference": reference,
            "prediction": prediction,
        })

    # 写出
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(results)} predictions -> {output_path}")


if __name__ == "__main__":
    main()
