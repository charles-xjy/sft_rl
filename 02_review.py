#!/usr/bin/env python3
"""
入口 02:人工审批前端。

启动本地 Web 服务,逐条对生成的 VQA 样本做通过/不通过判定。
审批结果写入 outputs/manual_reviews.jsonl(幂等覆盖,同一样本以最后一次为准)。
浏览器打开终端打印的 url 即可;Ctrl+C 停止。

逻辑在 src/review.py。
"""
import argparse

from src import review


def main():
    p = argparse.ArgumentParser(description="启动本地人工审批服务")
    p.add_argument("--input", default="outputs/03_answers_sft.jsonl", help="要审核的 SFT JSONL")
    p.add_argument("--reviews", default="outputs/manual_reviews.jsonl", help="审核结果输出 JSONL 路径")
    p.add_argument("--predictions", default=None,
                   help="学生模型预测 JSONL（05_baseline.py 风格 {image,question,prediction}）；"
                        "传入后前端可对比教师 vs 学生答案。如 outputs/baseline/val_pred.jsonl")
    p.add_argument("--host", default="127.0.0.1", help="监听地址；要别的机器访问改 0.0.0.0")
    p.add_argument("--port", type=int, default=8008, help="监听端口")
    args = p.parse_args()

    try:
        review.serve(args)
    except KeyboardInterrupt:
        print("\n[stopped]")


if __name__ == "__main__":
    main()
