"""
SFT data health report + optional rebalance for pure distillation data.

For the current setup, the health report focuses on:
- question distribution and collapse risk
- answer length and answer-opening template collapse
- light warning-only scans such as residual tags, risky words, and encoding anomalies

It intentionally does not evaluate refusal quality or teacher-side safety behavior.
"""

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict

from src.sampling import SCHEDULE_DIST
from src.validator import CONTEXTUAL_RISK_PATTERNS, HARD_RISK_WORDS

REFUSAL_TYPES = ("unanswerable", "ambiguous")
KEEP_ALL_TYPES = REFUSAL_TYPES

_MAIN = {k: v for k, v in SCHEDULE_DIST.items() if k not in REFUSAL_TYPES}
_MAIN_SUM = sum(_MAIN.values()) or 1.0
TARGET_DIST = {k: v / _MAIN_SUM for k, v in _MAIN.items()}

TAG_RE = re.compile(r"</?(analysis|answer)>")
LONG_WARN_P90 = 120


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _risk_hits(text):
    hits = []
    for w in HARD_RISK_WORDS:
        if w in text:
            hits.append(w)
    for pat in CONTEXTUAL_RISK_PATTERNS:
        if re.search(pat, text):
            hits.append("精确数值+单位")
            break
    return hits


def report(rows, top_n=12, threshold=0.20):
    """
    Print a compact health report.

    Returns True when any hard redline is triggered:
    - question-type entropy < 0.7
    - top-3 question prefixes > 60%
    - single answer opening > threshold
    """
    n = len(rows)
    if n == 0:
        print("数据为空")
        return False

    qtype = Counter()
    q_prefix = Counter()
    ans_open = Counter()
    len_by_type = defaultdict(list)
    all_len = []
    tag_residual = 0
    risk_rows = 0
    risk_words = Counter()
    cjk = cyr = total_chars = 0

    for row in rows:
        qt = row.get("meta", {}).get("question_type", "?")
        qtype[qt] += 1

        question = row["messages"][0]["content"].replace("<image>", "").strip()
        answer = row["messages"][1]["content"]

        q_prefix[question[:6]] += 1
        ans_open[answer[:12]] += 1
        len_by_type[qt].append(len(answer))
        all_len.append(len(answer))

        if TAG_RE.search(answer):
            tag_residual += 1

        hits = _risk_hits(answer)
        if hits:
            risk_rows += 1
            for w in hits:
                risk_words[w] += 1

        for ch in answer:
            total_chars += 1
            code = ord(ch)
            if 0x4E00 <= code <= 0x9FFF:
                cjk += 1
            elif 0x0400 <= code <= 0x04FF:
                cyr += 1

    red = False
    print(f"\n{'=' * 64}\n数据体检报告（纯蒸馏，共 {n} 条）\n{'=' * 64}")

    print("\n[1] question_type 配比 vs 目标（SCHEDULE_DIST）")
    for qt in SCHEDULE_DIST:
        cnt = qtype.get(qt, 0)
        tgt = SCHEDULE_DIST[qt]
        cur = cnt / n
        flag = "  <-- 偏高" if cur > tgt * 1.5 else ("  <-- 偏低" if cur < tgt * 0.5 else "")
        print(f"  {qt:16s} {cnt:5d}  当前 {cur:6.1%}  目标 {tgt:5.0%}{flag}")
    for qt in qtype:
        if qt not in SCHEDULE_DIST:
            print(f"  {qt:16s} {qtype[qt]:5d}  当前 {qtype[qt] / n:6.1%}  (非目标类型)")
    ent = _norm_entropy(qtype)
    print(
        f"  归一化熵 = {ent:.2f} (1=完全均匀，越低越偏；< 0.7 视为分布失衡)"
        + ("  <-- 偏低[红线]" if ent < 0.7 else "")
    )
    if ent < 0.7:
        red = True

    print(f"\n[2] 问题前缀 Top{top_n}  [红线: 前 3 前缀合计 > 60%]")
    top3 = sum(v for _, v in q_prefix.most_common(3)) / n
    for k, v in q_prefix.most_common(top_n):
        print(f"  {v / n:6.1%}  {k!r}")
    print(
        f"  前 3 前缀合计 = {top3:.1%}"
        + ("  <-- 塌缩[红线]" if top3 > 0.60 else "")
    )
    if top3 > 0.60:
        red = True

    print(f"\n[3] 答案开头 Top{top_n}  [红线: 单一开头 > {threshold:.0%}]")
    red |= _print_open(ans_open, n, top_n, threshold)

    print(f"\n[4] 答案长度（字符）  [告警: p90 > {LONG_WARN_P90}]")
    p90 = sorted(all_len)[min(n - 1, int(n * 0.9))]
    print(
        f"  总体: 平均 {statistics.mean(all_len):5.1f}  中位 {statistics.median(all_len):.0f}"
        f"  p90 {p90}  最长 {max(all_len)}"
        + ("  <-- 偏长，可能报告化" if p90 > LONG_WARN_P90 else "")
    )
    for qt, ls in sorted(len_by_type.items()):
        print(
            f"  {qt:16s} n={len(ls):5d}  平均 {statistics.mean(ls):5.1f}"
            f"  中位 {statistics.median(ls):5.0f}  最长 {max(ls):5d}"
        )

    print(f"\n[5] 残留 <analysis>/<answer> 标签: {tag_residual} 条")
    if tag_residual:
        print("  <-- 说明旧模板输出仍有残留（告警）")

    print(f"\n[6] 越界/高风险词命中: {risk_rows} 条 ({risk_rows / n:.1%})")
    for w, c in risk_words.most_common(top_n):
        print(f"  {c:5d}  {w}")

    cyr_rate = cyr / total_chars if total_chars else 0
    cjk_rate = cjk / total_chars if total_chars else 0
    print(
        f"\n[7] 编码自检: 中文 {cjk_rate:.0%}  西里尔字母 {cyr_rate:.2%}"
        + ("  <-- 可能有乱码" if cyr_rate > 0.005 else "")
    )

    print(f"\n{'=' * 64}")
    print("结论: " + ("[!] 触发红线，建议处理后再发布" if red else "[OK] 未触发红线"))
    print("=" * 64)
    return red


def _norm_entropy(counter):
    total = sum(counter.values())
    if total == 0 or len(counter) <= 1:
        return 1.0
    h = -sum((c / total) * math.log(c / total) for c in counter.values() if c)
    return h / math.log(len(counter))


def _print_open(counter, total, top_n, threshold):
    red = False
    for k, v in counter.most_common(top_n):
        frac = v / total if total else 0
        flag = "  <-- 超红线" if frac > threshold else ""
        if flag:
            red = True
        print(f"  {frac:6.1%}  {k!r}{flag}")
    return red


def rebalance_rows(rows, seed=42):
    """Rebalance major answerable types; keep refusal-like question types intact."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("meta", {}).get("question_type", "?")].append(row)

    n_cap = min(
        (len(buckets.get(t, [])) / w for t, w in TARGET_DIST.items() if w > 0),
        default=0,
    )
    n_cap = int(n_cap)

    selected = []
    print(f"\n降采样目标总量（主类型）≈ {n_cap} 条:")
    for t, w in TARGET_DIST.items():
        avail = buckets.get(t, [])
        take = min(int(round(w * n_cap)), len(avail))
        picked = rng.sample(avail, take) if take < len(avail) else list(avail)
        selected.extend(picked)
        print(f"  {t:16s} 可用 {len(avail):5d} -> 取 {take:5d} ({take / max(n_cap, 1):.0%})")

    for t in KEEP_ALL_TYPES:
        kept = buckets.get(t, [])
        selected.extend(kept)
        if kept:
            print(f"  {t:16s} 可用 {len(kept):5d} -> 取 {len(kept):5d} (全保留)")

    rng.shuffle(selected)
    return selected


def rebalance(rows, out_path, seed=42):
    selected = rebalance_rows(rows, seed)
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[done] 发布集 {len(selected)} 条 -> {out_path}")
    return selected
