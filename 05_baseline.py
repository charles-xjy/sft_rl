#!/usr/bin/env python3
# ============================================================================
# 基线答案生成：用「未微调」的原始 Qwen3-VL-2B 跑一遍测试集，产出 baseline 预测。
#
#   目的：给微调后的模型一个对照。同一批 (图像 + 问题) 同时喂给基线模型，
#   把它的回答存下来，之后和微调模型 / 教师参考答案并排比较。
#
#   关键设计：发给基线的是「裸问题」(只有 question 文本 + 图)，不是数据生成时
#   那个 ANSWER_DISTILL_PROMPT。因为微调学生训练时的 user 输入就是裸问题
#   (见 04_train.py 的 convert_to_conversation)，基线必须吃到完全相同的输入，
#   对照才公平。distill prompt 只在「造教师标签」时用，与推理输入无关。
#
#   读 03_convert.py 产出的 val.jsonl（或任意 {image,question,answer} jsonl）。
#   支持并发 + 断点续跑（按 image|question 去重）。
# ============================================================================
import argparse
import json
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

from src.vlm_client import VLMClient


def parse_args():
    p = argparse.ArgumentParser(description="未微调基线模型答案生成")
    p.add_argument("--src", type=str, default="outputs/sft/val.jsonl",
                   help="测试集 jsonl（每行 {image, question, answer, ...}）")
    p.add_argument("--out", type=str, default="outputs/baseline/val_pred.jsonl",
                   help="基线预测输出 jsonl")
    p.add_argument("--base-url", type=str, default="http://10.129.107.145:8002/v1",
                   help="基线模型 vLLM 服务地址（未微调原始模型，端口 8002）")
    p.add_argument("--model", type=str, default=None,
                   help="模型名；默认自动探测服务端实际加载的模型")
    p.add_argument("--max-concurrency", type=int, default=16, help="并发请求数")
    p.add_argument("--retries", type=int, default=3, help="单条失败最大重试次数")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="单条回答最大 token（基线可能啰嗦，给足空间避免截断）")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="采样温度；基线对照默认 0.0 走贪心，结果可复现")
    p.add_argument("--limit", type=int, default=None,
                   help="只跑前 N 条（快速冒烟用）")
    return p.parse_args()


def load_rows(src: Path):
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_keys(out_path: Path) -> set:
    """已完成的 image|question 键，用于断点续跑。"""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(f"{r['image']}|{r['question']}")
            except Exception:
                pass
    return done


def make_client_factory(base_url: str, model: str, retries: int):
    """每线程一个 VLMClient（OpenAI 客户端按线程隔离，稳妥）。"""
    local = threading.local()

    def get_client() -> VLMClient:
        c = getattr(local, "client", None)
        if c is None:
            c = VLMClient(base_url=base_url, model=model, retries=retries)
            local.client = c
        return c
    return get_client


def worker(row: dict, get_client, max_tokens: int, temperature: float) -> dict:
    """对一条样本：裸问题 + 图 → 基线模型回答。"""
    client = get_client()
    image_path = Path(row["image"])
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    prediction = client.call(
        prompt=row["question"],          # ← 裸问题，与微调学生的推理输入一致
        image_path=image_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "image": row["image"],
        "question": row["question"],
        "question_type": row.get("question_type", ""),
        "reference": row.get("answer", ""),   # 教师参考答案（val 里带的）
        "prediction": prediction,             # 未微调基线的回答
    }


def append_jsonl(path: Path, record: dict, lock: threading.Lock):
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    src = Path(args.src)
    out_path = Path(args.out)

    rows = load_rows(src)
    if args.limit is not None:
        rows = rows[:args.limit]

    done = load_done_keys(out_path)
    pending = [r for r in rows if f"{r['image']}|{r['question']}" not in done]

    print("=" * 64)
    print("未微调基线答案生成")
    print("=" * 64)
    print(f"测试集     : {src} （{len(rows)} 条）")
    print(f"已完成跳过 : {len(done)} 条")
    print(f"本次待跑   : {len(pending)} 条")
    print(f"基线服务   : {args.base_url}")
    print(f"输出       : {out_path}")
    print(f"并发 / 温度: {args.max_concurrency} / {args.temperature}")
    print("=" * 64)
    if not pending:
        print("没有待处理样本，结束。")
        return

    # 先建一个客户端触发模型自动探测，打印一次实际模型名
    bootstrap = VLMClient(base_url=args.base_url, model=args.model, retries=args.retries)
    get_client = make_client_factory(bootstrap.base_url, bootstrap.model, args.retries)

    out_lock = threading.Lock()
    stats = {"ok": 0, "err": 0}

    queue = list(pending)
    future_to_row = {}

    with tqdm(total=len(pending), desc="baseline", position=0) as bar, \
         ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:

        def submit_some():
            while queue and len(future_to_row) < args.max_concurrency:
                r = queue.pop()
                fut = ex.submit(worker, r, get_client, args.max_tokens, args.temperature)
                future_to_row[fut] = r

        submit_some()
        while future_to_row:
            done_futs, _ = wait(list(future_to_row.keys()), return_when=FIRST_COMPLETED)
            for fut in done_futs:
                r = future_to_row.pop(fut)
                try:
                    record = fut.result()
                    append_jsonl(out_path, record, out_lock)
                    stats["ok"] += 1
                except Exception as e:
                    stats["err"] += 1
                    tqdm.write(f"[error] {r.get('image','')}: {type(e).__name__}: {e}")
                bar.update(1)
                bar.set_postfix(ok=stats["ok"], err=stats["err"])
            submit_some()

    print("\n" + "=" * 64)
    print(f"完成：成功 {stats['ok']} / 失败 {stats['err']} -> {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
