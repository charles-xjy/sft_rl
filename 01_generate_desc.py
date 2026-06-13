#!/usr/bin/env python3
"""
Phase 1: 生成受控图像描述

输入: 遥感图像
输出: outputs/01_descriptions.jsonl
格式: {image_path, dataset, description}
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from src.scanner import scan_all_datasets, stratified_sample, load_processed_records
from src.vlm_client import VLMClient
from src.prompts import DESCRIPTION_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Phase 1: 生成受控图像描述")
    parser.add_argument("--dataset-root", type=str, default="dataset",
                        help="数据集根目录")
    parser.add_argument("--datasets", type=str, default="EBD,LEVIR-CD+,SECOND",
                        help="要处理的数据集，逗号分隔")
    parser.add_argument("--num-images", type=int, default=100,
                        help="处理的图像数量")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称")
    parser.add_argument("--base-url", type=str, default="http://10.129.107.145:8001/v1",
                        help="vLLM服务地址")
    parser.add_argument("--retries", type=int, default=3,
                        help="失败重试次数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")

    args = parser.parse_args()

    # 输出文件
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path("outputs/01_descriptions.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 扫描图像
    dataset_root = Path(args.dataset_root)
    datasets = [d.strip() for d in args.datasets.split(",")]
    images_by_dataset = scan_all_datasets(dataset_root, datasets)

    print("="*60)
    print("Phase 1: 生成受控图像描述")
    print("="*60)
    for dataset, images in images_by_dataset.items():
        print(f"[{dataset}] 找到 {len(images)} 张图像")

    # 分层采样
    samples = stratified_sample(images_by_dataset, args.num_images, args.seed)
    print(f"\n分层采样后共选择 {len(samples)} 张图像")

    # 断点续传
    processed_images = load_processed_records(output_file, "image_path")
    if processed_images:
        print(f"发现已处理 {len(processed_images)} 张图像，跳过...")
        samples = [(d, p) for d, p in samples if str(p.resolve()) not in processed_images]
        print(f"剩余待处理: {len(samples)} 张图像")

    if not samples:
        print("\n所有图像已处理完成！")
        return

    # 初始化VLM客户端
    vlm = VLMClient(base_url=args.base_url, model=args.model, retries=args.retries)

    # 主处理循环
    success_count = 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        pbar = tqdm(samples)
        for dataset, image_path in pbar:
            pbar.set_description(f"处理中: {dataset}")

            try:
                description = vlm.call(
                    prompt=DESCRIPTION_PROMPT,
                    image_path=image_path,
                    max_tokens=512,
                    temperature=0.3
                )

                if description:
                    record = {
                        "image_path": str(image_path.resolve()),
                        "dataset": dataset,
                        "description": description,
                        "generated_at": datetime.now().isoformat()
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    success_count += 1

                pbar.set_postfix({"成功": success_count})

            except Exception as e:
                print(f"\n[错误] {image_path.name}: {e}")
                continue

    print("\n" + "="*60)
    print("Phase 1 完成！")
    print(f"输出文件: {output_file}")
    print(f"成功生成描述: {success_count} 张图像")
    print("="*60)


if __name__ == "__main__":
    main()
