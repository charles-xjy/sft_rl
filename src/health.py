"""
SFT 数据体检 + 按配比降采样。

把"是否过度模板化、配比是否失衡"量化出来,作为数据发布前的最后一道闸。
`report()` 只读,返回是否触发红线;`rebalance()` 按目标配比降采样产出发布集。

被 `01_generate.py`(生成后自动体检)与 `03_convert.py` 复用;只依赖标准库。
"""
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict

# 目标 question_type 配比(与 prompts.py 的 Phase2 建议一致)
TARGET_DIST = {
    "existence": 0.35,
    "location": 0.25,
    "spatial_relation": 0.20,
    "attribute": 0.15,
    "counting": 0.05,
}
# unanswerable / ambiguous 数量少且珍贵,降采样时全部保留,不纳入主配比
KEEP_ALL_TYPES = ("unanswerable", "ambiguous")
REFUSAL_TYPES = ("unanswerable", "ambiguous")

REFUSAL_RE = re.compile(r"无法|不确定|难以确认|不能确定|可能是.*也可能")


def extract(tag, s):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", s, re.S)
    return m.group(1).strip() if m else ""


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def report(rows, top_n=12, threshold=0.20):
    """打印体检报告,返回是否触发红线(True=有问题)。"""
    n = len(rows)
    ans_open = Counter()
    ana_open = Counter()
    q_prefix = Counter()      # 问题前缀(塌缩监控)
    qtype = Counter()
    mode = Counter()
    len_by_type = defaultdict(list)
    refusal_total = refusal_ok = 0
    cjk = cyr = total_chars = 0

    for o in rows:
        meta = o.get("meta", {})
        qt = meta.get("question_type", "?")
        am = meta.get("answer_mode") or ("analysis" if "<analysis>" in o["messages"][1]["content"] else "direct")
        qtype[qt] += 1
        mode[am] += 1

        content = o["messages"][1]["content"]
        question = o["messages"][0]["content"].replace("<image>", "").strip()
        answer = extract("answer", content)
        analysis = extract("analysis", content)
        ans_open[answer[:12]] += 1
        q_prefix[question[:6]] += 1
        if analysis:
            ana_open[analysis.lstrip()[:14]] += 1
        len_by_type[qt].append(len(answer))

        if qt in REFUSAL_TYPES:
            refusal_total += 1
            if REFUSAL_RE.search(answer):
                refusal_ok += 1

        for ch in content:
            total_chars += 1
            c = ord(ch)
            if 0x4E00 <= c <= 0x9FFF:
                cjk += 1
            elif 0x0400 <= c <= 0x04FF:
                cyr += 1

    red = False
    print(f"\n{'='*64}\n数据体检报告  共 {n} 条\n{'='*64}")

    # ── 问题侧:分布塌缩监控(reviewer 认为最危险,放最前) ──
    # [1] question_type 配比 + 归一化熵
    print(f"\n[1] question_type 配比 vs 目标（含归一化熵）")
    types_all = list(TARGET_DIST) + list(KEEP_ALL_TYPES)
    for qt in types_all:
        cnt = qtype.get(qt, 0)
        tgt = TARGET_DIST.get(qt)
        cur = cnt / n if n else 0
        tgt_s = f"目标 {tgt:.0%}" if tgt is not None else "目标 少量"
        flag = "  <-- 偏高" if (tgt is not None and cur > tgt * 2) else ""
        print(f"  {qt:16s} {cnt:5d}  当前 {cur:6.1%}  {tgt_s}{flag}")
    ent = _norm_entropy(qtype)
    print(f"  归一化熵 = {ent:.2f}（1=完全均匀，越低越偏；< 0.7 视为类型分布失衡）"
          + ("  <-- 偏低" if ent < 0.7 else ""))
    if ent < 0.7:
        red = True

    # [2] 问题前缀塌缩(Question Distribution Collapse)
    print(f"\n[2] 问题前缀 Top{top_n}（红线: 前 3 前缀合计 > 60%）")
    top3 = sum(v for _, v in q_prefix.most_common(3)) / n if n else 0
    for k, v in q_prefix.most_common(top_n):
        print(f"  {v/n:6.1%}  {k!r}")
    print(f"  前 3 前缀合计 = {top3:.1%}" + ("  <-- 塌缩!问题多样性不足" if top3 > 0.60 else ""))
    if top3 > 0.60:
        red = True

    # ── 答案侧:过度模板化监控 ──
    print(f"\n[3] <answer> 开头 Top{top_n}（红线: 单一开头 > {threshold:.0%}）")
    red |= _print_open(ans_open, n, top_n, threshold)
    print(f"\n[4] <analysis> 开头 Top{top_n}（红线: 单一开头 > {threshold:.0%}）")
    red |= _print_open(ana_open, sum(ana_open.values()), top_n, threshold)

    # [5] answer_mode 比例(是否达标 analysis/direct)
    print(f"\n[5] answer_mode 比例")
    for m, c in mode.most_common():
        print(f"  {m:10s} {c:5d}  {c/n:6.1%}")
    if mode.get("direct", 0) == 0:
        print("  <-- 警告: 没有 direct 直答样本,全部带 analysis,过度模板化风险高")
        red = True

    # [6] 各题型 answer 长度
    print(f"\n[6] 各题型 <answer> 长度（平均 / 中位）")
    for qt, ls in sorted(len_by_type.items()):
        print(f"  {qt:16s} n={len(ls):5d}  平均 {statistics.mean(ls):5.1f}  中位 {statistics.median(ls):.0f}")

    # [7] 拒答率(目标 5%-10%) + 编码
    print(f"\n[7] 其他")
    refusal_share = refusal_total / n if n else 0
    band = "" if 0.05 <= refusal_share <= 0.10 else "  <-- 不在 5%-10% 目标带"
    print(f"  拒答类(unanswerable+ambiguous)占比: {refusal_total}/{n} = {refusal_share:.1%}{band}")
    if refusal_total:
        rate = refusal_ok / refusal_total
        print(f"  拒答类正确给出不确定/多解表达: {refusal_ok}/{refusal_total} = {rate:.0%}"
              + ("  <-- 偏低,部分拒答题被强答" if rate < 0.9 else ""))
    else:
        print("  拒答类: 0 条（建议构造 5%-10%）")
    cyr_rate = cyr / total_chars if total_chars else 0
    print(f"  编码自检: 中文 {cjk/total_chars:.0%}  西里尔(乱码嫌疑) {cyr_rate:.2%}"
          + ("  <-- 注意可能乱码" if cyr_rate > 0.005 else "  (正常)"))

    print(f"\n{'='*64}")
    print("结论: " + ("[!] 触发红线,建议处理后再发布" if red else "[OK] 未触发红线"))
    print('='*64)
    return red


def _norm_entropy(counter):
    """问题类型分布的归一化香农熵(0~1,1=完全均匀)。"""
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
    """按 TARGET_DIST 对主类型降采样;unanswerable/ambiguous 全保留。返回挑选后的样本列表。"""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for o in rows:
        buckets[o.get("meta", {}).get("question_type", "?")].append(o)

    # 以最紧的桶为基准确定总量 N: N = min(可用数 / 目标占比)
    n_cap = min(
        (len(buckets.get(t, [])) / w for t, w in TARGET_DIST.items() if w > 0),
        default=0,
    )
    n_cap = int(n_cap)

    selected = []
    print(f"\n降采样(目标总量主类型≈{n_cap} 条):")
    for t, w in TARGET_DIST.items():
        avail = buckets.get(t, [])
        take = min(int(round(w * n_cap)), len(avail))
        picked = rng.sample(avail, take) if take < len(avail) else list(avail)
        selected.extend(picked)
        print(f"  {t:16s} 可用 {len(avail):5d} -> 取 {take:5d} ({take/max(n_cap,1):.0%})")

    for t in KEEP_ALL_TYPES:
        kept = buckets.get(t, [])
        selected.extend(kept)
        if kept:
            print(f"  {t:16s} 可用 {len(kept):5d} -> 取 {len(kept):5d} (全保留)")

    rng.shuffle(selected)
    return selected


def rebalance(rows, out_path, seed=42):
    """按 TARGET_DIST 降采样并写入 out_path(发布集)。"""
    selected = rebalance_rows(rows, seed)
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for o in selected:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"\n[done] 发布集 {len(selected)} 条 -> {out_path}")
    return selected
