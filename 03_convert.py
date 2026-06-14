#!/usr/bin/env python3
"""
入口 03:把 SFT 答案转成 Unsloth 视觉训练集,并过滤未过质检的样本。

做的事:过滤(默认只留 pass,永远剔除 format_fail) -> 去 <image> 占位符 ->
图片路径根目录重映射(始终输出 POSIX) -> 可选校验图片存在 -> 切分 train/val。
产出 outputs/sft/train.jsonl 与 val.jsonl,供 04_train.py 直接读取。

逻辑在 src/convert.py。本脚本不依赖深度学习库,Windows 上也能跑
(除非开 --check-images 需要图片在本机)。
"""
import argparse

from src import convert


def main():
    p = argparse.ArgumentParser(description="转 Unsloth 训练集 + 过滤未过质检样本")
    p.add_argument("--src", default="outputs/03_answers_sft.jsonl", help="输入答案 jsonl(也可指向降采样后的发布集)")
    p.add_argument("--out-dir", default="outputs/sft", help="输出目录")
    p.add_argument("--old-root", default=convert.DEFAULT_OLD_ROOT, help="原图片根目录(被替换)")
    p.add_argument("--image-root", default="", help="训练机上的图片根目录(替换成它);留空则不改路径")
    p.add_argument("--keep-warning", action="store_true", help="也保留 warning 样本(默认只留 pass，永远剔除 format_fail)")
    p.add_argument("--rebalance", action="store_true", help="按目标配比降采样(主要把 counting 压到 5%%)后再转换")
    p.add_argument("--check-images", action="store_true", help="校验每张图片是否存在(需图片在本机)")
    p.add_argument("--val-ratio", type=float, default=0.02, help="验证集比例")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    convert.prepare(args)


if __name__ == "__main__":
    main()
