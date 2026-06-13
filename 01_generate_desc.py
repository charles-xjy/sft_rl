#!/usr/bin/env python3
"""
Phase 1: generate controlled image descriptions.

Input: remote sensing images
Output: outputs/01_descriptions.jsonl
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.prompts import DESCRIPTION_PROMPT
from src.scanner import (
    load_manifest,
    load_processed_records,
    sample_per_dataset,
    save_manifest,
    scan_all_datasets,
    stratified_sample,
)
from src.vlm_client import VLMClient


def main():
    parser = argparse.ArgumentParser(description="Phase 1: 生成图像描述")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="数据集根目录")
    parser.add_argument("--datasets", type=str, default="LEVIR-CD+,SECOND", help="要处理的数据集，逗号分隔")
    parser.add_argument("--num-images", type=int, default=None, help="全局随机采样总数；传入后优先于 --samples-per-dataset")
    parser.add_argument("--samples-per-dataset", type=int, default=10, help="每个数据集采样数量")
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--base-url", type=str, default="http://10.129.107.145:8001/v1", help="vLLM 服务地址")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--manifest", type=str, default=None, help="样本清单文件路径；默认 outputs/sample_manifest.jsonl")
    parser.add_argument("--refresh-manifest", action="store_true", help="忽略已有 manifest，重新采样并覆盖")
    args = parser.parse_args()

    output_file = Path(args.output) if args.output else Path("outputs/01_descriptions.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_file.parent / "sample_manifest.jsonl"

    dataset_root = Path(args.dataset_root)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    images_by_dataset = scan_all_datasets(dataset_root, datasets)

    print("=" * 60)
    print("Phase 1: 生成图像描述")
    print("=" * 60)
    for dataset, images in images_by_dataset.items():
        print(f"[{dataset}] 找到 {len(images)} 张图像")

    if manifest_path.exists() and not args.refresh_manifest:
        samples = load_manifest(manifest_path)
        print(f"\n从 manifest 载入样本: {manifest_path} ({len(samples)} 张)")
    else:
        if args.num_images is not None:
            samples = stratified_sample(images_by_dataset, args.num_images, args.seed)
            print(f"\n全局随机采样后共选择 {len(samples)} 张图像")
        else:
            samples = sample_per_dataset(images_by_dataset, args.samples_per_dataset, args.seed)
            print(f"\n按数据集采样：每个数据集 {args.samples_per_dataset} 张，共 {len(samples)} 张图像")
        save_manifest(manifest_path, samples)
        print(f"已写入 manifest: {manifest_path}")

    processed_images = load_processed_records(output_file, "image_path")
    if processed_images:
        print(f"发现已处理 {len(processed_images)} 张图像，跳过")
        samples = [(d, p) for d, p in samples if str(p.resolve()) not in processed_images]
        print(f"剩余待处理 {len(samples)} 张图像")

    if not samples:
        print("\n所有图像已处理完成")
        return

    vlm = VLMClient(base_url=args.base_url, model=args.model, retries=args.retries)

    success_count = 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        pbar = tqdm(samples)
        for dataset, image_path in pbar:
            pbar.set_description(f"处理中 {dataset}")
            try:
                description = vlm.call(
                    prompt=DESCRIPTION_PROMPT,
                    image_path=image_path,
                    max_tokens=4096,
                    temperature=0.3,
                )

                if description:
                    record = {
                        "image_path": str(image_path.resolve()),
                        "dataset": dataset,
                        "description": description,
                        "generated_at": datetime.now().isoformat(),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    success_count += 1
                else:
                    print(f"\n[empty-output] {image_path.name}: model returned empty description")

                pbar.set_postfix({"success": success_count})
            except Exception as e:
                print(f"\n[error] {image_path.name}: {e}")
                continue

    print("\n" + "=" * 60)
    print("Phase 1 完成")
    print(f"输出文件: {output_file}")
    print(f"成功生成描述: {success_count} 张图像")
    print("=" * 60)


if __name__ == "__main__":
    main()
