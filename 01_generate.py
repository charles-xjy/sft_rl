#!/usr/bin/env python3
"""
入口 01:生成基础 SFT 数据 + 数据质量分析报告。

三阶段并行生成(描述 -> 出题 -> 带分析答案),产出:
  outputs/01_descriptions.jsonl / 02_questions.jsonl / 03_answers_sft.jsonl
  outputs/03_stats.json / sample_manifest.jsonl
生成结束后自动跑一次只读体检(配比/塌缩/拒答率/编码 + 红线告警);
触发红线只提示、不中断,降采样请在人工抽检后用 03_convert.py 进行。

逻辑在 src/generation.py 与 src/health.py。
"""
import argparse

from src import generation, health


def main():
    p = argparse.ArgumentParser(description="生成基础 SFT 数据并给出质量分析报告")
    p.add_argument("--dataset-root", type=str, default="dataset", help="数据集根目录")
    p.add_argument("--datasets", type=str, default="LEVIR-CD+,SECOND", help="要处理的数据集，逗号分隔")
    p.add_argument("--num-images", type=int, default=None, help="全局随机采样总数；传入后优先于 --samples-per-dataset")
    p.add_argument(
        "--samples-per-dataset",
        type=str,
        default="LEVIR-CD+=500,SECOND=500",
        help=(
            "每个数据集采样数量。两种语法："
            "(1) 单一整数 '10' — 每个数据集都抽 10 张；"
            "(2) 按数据集指定 'LEVIR-CD+=500,SECOND=500' — 仅抽指定的"
        ),
    )
    p.add_argument(
        "--ebd-per-disaster",
        type=int,
        default=None,
        help="EBD 按 7 种灾害子目录各抽 N 张；启用后会覆盖 --samples-per-dataset 里 EBD 的值",
    )
    p.add_argument("--num-questions", type=int, default=5, help="每张图最多生成问题数量")
    p.add_argument("--min-questions", type=int, default=2, help="每张图最少生成问题数量")
    p.add_argument("--max-concurrency", type=int, default=24, help="整体最大并发请求数")
    p.add_argument("--model", type=str, default=None, help="模型名称")
    p.add_argument("--base-url", type=str, default="http://10.129.107.145:8001/v1", help="vLLM 服务地址")
    p.add_argument("--retries", type=int, default=3, help="失败重试次数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--direct-ratio", type=float, default=0.3,
                   help="不带 analysis 的直答样本占比(打散固定起手式,缓解过度模板化)")
    p.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    p.add_argument("--manifest", type=str, default=None, help="样本清单文件路径；默认 outputs/sample_manifest.jsonl")
    p.add_argument("--refresh-manifest", action="store_true", help="忽略已有 manifest，重新采样并覆盖")
    p.add_argument("--no-health-check", action="store_true", help="跳过生成结束后的自动数据体检")
    args = p.parse_args()

    answer_path = generation.generate(args)

    if not args.no_health_check and answer_path.exists():
        print("\n" + "=" * 80)
        print("自动体检(只读)。降采样请在人工抽检后用 03_convert.py 处理。")
        print("=" * 80, flush=True)
        rows = health.load(answer_path)
        red = health.report(rows)
        if red:
            print("\n[health-check] [!] 体检触发红线,请按上方报告处理后再发布。")


if __name__ == "__main__":
    main()
