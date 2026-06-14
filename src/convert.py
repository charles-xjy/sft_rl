"""
把生成的 SFT 答案(03_answers_sft.jsonl 风格)转成 Unsloth 视觉微调训练集。

做的事:
  1. 过滤:默认只保留自动质检 pass 的样本(keep_warning 可放宽,永远剔除 format_fail)。
  2. 去掉 user 文本里的 `<image>\n` 占位符(图像走结构化 content,不靠占位符)。
  3. 图片路径根目录重映射:生成时写死的路径 -> 训练机真实根目录(始终输出 POSIX)。
  4. 可选校验每张图片是否存在。
  5. 按比例切分 train / val,输出归一化 jsonl。

入口脚本 `03_convert.py` 解析命令行后调用 `prepare(args)`。
本模块只依赖标准库,无图片/深度学习依赖。
"""
import json
import random
from pathlib import Path, PurePosixPath

# 生成阶段写死在 images 字段里的根目录(见 03_answers_sft.jsonl)。
DEFAULT_OLD_ROOT = "/home/charles/mycode/sft+rl/dataset"
IMAGE_TOKEN = "<image>"


def remap_image(path: str, old_root: str, new_root: str) -> str:
    """把图片路径的根目录从 old_root 换成 new_root,保留其后的相对结构。

    始终用 POSIX 分隔符输出,这样即便在 Windows 上跑本脚本,产出的路径
    在 Linux 训练机上也直接可用。
    """
    if not new_root:
        return path
    op = PurePosixPath(old_root)
    pp = PurePosixPath(path.replace("\\", "/"))
    try:
        rel = pp.relative_to(op)
    except ValueError:
        # 路径不在预期根目录下,原样返回并交给 check_images 暴露。
        return path
    return str(PurePosixPath(new_root.replace("\\", "/")) / rel)


def strip_image_token(text: str) -> str:
    """移除 `<image>` 占位符及其后紧跟的换行,返回纯问题文本。"""
    out = text.replace(IMAGE_TOKEN + "\n", "").replace(IMAGE_TOKEN, "")
    return out.strip()


def load_records(src: Path, keep_warning: bool):
    kept, dropped = [], 0
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            av = r.get("meta", {}).get("auto_validation", {})
            label = av.get("label", "pass")
            ok = av.get("is_valid", True) and (label == "pass" or (keep_warning and label != "format_fail"))
            if not ok:
                dropped += 1
                continue
            kept.append(r)
    return kept, dropped


def to_train_record(r: dict, old_root: str, new_root: str) -> dict:
    msgs = r["messages"]
    user = next(m for m in msgs if m["role"] == "user")
    assistant = next(m for m in msgs if m["role"] == "assistant")
    image = remap_image(r["images"][0], old_root, new_root)
    return {
        "image": image,
        "question": strip_image_token(user["content"]),
        "answer": assistant["content"].strip(),
        "question_type": r["meta"].get("question_type", ""),
        "answer_mode": r["meta"].get("answer_mode", "analysis"),
    }


def prepare(args) -> None:
    """把 SFT 答案转成 Unsloth 训练集。args 为命令行解析后的 Namespace。"""
    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, dropped = load_records(src, args.keep_warning)

    rebalanced = 0
    if getattr(args, "rebalance", False):
        from src import health
        before = len(records)
        records = health.rebalance_rows(records, args.seed)
        rebalanced = before - len(records)

    train_records = [to_train_record(r, args.old_root, args.image_root) for r in records]

    missing = []
    if args.check_images:
        for tr in train_records:
            if not Path(tr["image"]).exists():
                missing.append(tr["image"])

    random.Random(args.seed).shuffle(train_records)
    n_val = int(len(train_records) * args.val_ratio)
    val, train = train_records[:n_val], train_records[n_val:]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    for path, rows in [(train_path, train), (val_path, val)]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 题型分布,方便确认配比。
    dist = {}
    for tr in train_records:
        dist[tr["question_type"]] = dist.get(tr["question_type"], 0) + 1

    print("=" * 60)
    print(f"输入 {src}")
    print(f"保留 {len(train_records)} 条,丢弃(未过滤通过) {dropped} 条"
          + (f",降采样剔除 {rebalanced} 条" if rebalanced else ""))
    print(f"切分: train {len(train)} / val {len(val)} (val_ratio={args.val_ratio})")
    print(f"题型分布: {dict(sorted(dist.items(), key=lambda x: -x[1]))}")
    if args.image_root:
        print(f"图片根目录: {args.old_root}  ->  {args.image_root}")
    else:
        print("图片路径未改(--image-root 为空)。Linux 上若路径不符,务必重跑并传 --image-root。")
    if args.check_images:
        print(f"图片存在性: 缺失 {len(missing)} 张" + (f",例: {missing[:3]}" if missing else " (全部就位)"))
    print(f"输出: {train_path}  /  {val_path}")
    print("=" * 60)
