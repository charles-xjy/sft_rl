"""
SFT 数据体检 + 按配比降采样(纯蒸馏版)。

纯蒸馏没有固定格式可校验,体检职责从"格式合规审计"转为"分布塌缩 + 内容风险"。
所有答案侧统计都跑在**完整答案文本**上(不再抠 <answer>/<analysis> 标签)。

`report()` 只读,返回是否触发红线(仅 3 条硬红线:类型熵 / 问题前缀塌缩 / 答案开头塌缩);
其余(啰嗦化、越界词、残留标签、拒答正确率、编码)只告警,不卡红线——纯蒸馏靠人工抽检兜底。
`rebalance()` 按目标配比降采样产出发布集。

被 `01_generate.py`(生成后自动体检)与 `03_convert.py` 复用。
目标分布对齐 `src/sampling.py:SCHEDULE_DIST`(单一真相源);越界词表复用 `src/validator.py`。
"""
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict

from src.sampling import SCHEDULE_DIST
from src.validator import CONTEXTUAL_RISK_PATTERNS, HARD_RISK_WORDS

# 拒答类:数量少、珍贵,降采样时全部保留,不纳入主配比
REFUSAL_TYPES = ("unanswerable", "ambiguous")
KEEP_ALL_TYPES = REFUSAL_TYPES

# 降采样用的主类型配比 = SCHEDULE_DIST 去掉拒答类后归一化到 1.0
_MAIN = {k: v for k, v in SCHEDULE_DIST.items() if k not in REFUSAL_TYPES}
_MAIN_SUM = sum(_MAIN.values()) or 1.0
TARGET_DIST = {k: v / _MAIN_SUM for k, v in _MAIN.items()}

# 拒答/不确定表达(含教师护栏用语"看不清/无法确认")
REFUSAL_RE = re.compile(r"无法|不确定|难以确认|不能确定|看不清|可能是.*也可能")
# 残留格式标签(纯蒸馏后应为 0)
TAG_RE = re.compile(r"</?(analysis|answer)>")
# 答案啰嗦化阈值(字符):p90 超过即提示"可能报告化"
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
    """返回命中的高风险词/模式(只读扫描,不阻断;纯蒸馏无格式校验,这里替你过一眼)。"""
    hits = []
    for w in HARD_RISK_WORDS:
        if w in text:
            hits.append(w)
    for pat in CONTEXTUAL_RISK_PATTERNS:
        if re.search(pat, text):
            hits.append("精确数值/单位")
            break
    return hits


def report(rows, top_n=12, threshold=0.20):
    """打印体检报告,返回是否触发红线(True=有问题)。

    硬红线只 3 条:类型熵 < 0.7、问题前缀 top3 > 60%、答案开头单一 > threshold。
    其余项只告警,不影响返回值。
    """
    n = len(rows)
    if n == 0:
        print("数据为空")
        return False

    qtype = Counter()
    q_prefix = Counter()       # 问题前缀(塌缩监控)
    ans_open = Counter()       # 答案开头(塌缩监控,跑在完整答案上)
    len_by_type = defaultdict(list)
    all_len = []
    refusal_total = refusal_ok = 0
    tag_residual = 0
    risk_rows = 0
    risk_words = Counter()
    cjk = cyr = total_chars = 0

    for o in rows:
        qt = o.get("meta", {}).get("question_type", "?")
        qtype[qt] += 1

        answer = o["messages"][1]["content"]          # 纯蒸馏:整条就是答案,无标签
        question = o["messages"][0]["content"].replace("<image>", "").strip()

        q_prefix[question[:6]] += 1
        ans_open[answer[:12]] += 1
        len_by_type[qt].append(len(answer))
        all_len.append(len(answer))

        if qt in REFUSAL_TYPES:
            refusal_total += 1
            if REFUSAL_RE.search(answer):
                refusal_ok += 1

        if TAG_RE.search(answer):
            tag_residual += 1

        hits = _risk_hits(answer)
        if hits:
            risk_rows += 1
            for w in hits:
                risk_words[w] += 1

        for ch in answer:
            total_chars += 1
            c = ord(ch)
            if 0x4E00 <= c <= 0x9FFF:
                cjk += 1
            elif 0x0400 <= c <= 0x04FF:
                cyr += 1

    red = False
    print(f"\n{'='*64}\n数据体检报告(纯蒸馏)  共 {n} 条\n{'='*64}")

    # ── 问题侧:分布塌缩(最危险,放最前;与答案格式无关,始终有效) ──
    # [1] question_type 配比 + 归一化熵  [红线: 熵 < 0.7]
    print("\n[1] question_type 配比 vs 目标(SCHEDULE_DIST) + 归一化熵")
    for qt in SCHEDULE_DIST:
        cnt = qtype.get(qt, 0)
        tgt = SCHEDULE_DIST[qt]
        cur = cnt / n
        flag = "  <-- 偏高" if cur > tgt * 1.5 else ("  <-- 偏低" if cur < tgt * 0.5 else "")
        print(f"  {qt:16s} {cnt:5d}  当前 {cur:6.1%}  目标 {tgt:5.0%}{flag}")
    for qt in qtype:
        if qt not in SCHEDULE_DIST:
            print(f"  {qt:16s} {qtype[qt]:5d}  当前 {qtype[qt]/n:6.1%}  (非目标类型)")
    ent = _norm_entropy(qtype)
    print(f"  归一化熵 = {ent:.2f}(1=完全均匀,越低越偏;< 0.7 视为类型分布失衡)"
          + ("  <-- 偏低[红线]" if ent < 0.7 else ""))
    if ent < 0.7:
        red = True

    # [2] 问题前缀塌缩(Question Distribution Collapse)  [红线: top3 > 60%]
    print(f"\n[2] 问题前缀 Top{top_n}  [红线: 前 3 前缀合计 > 60%]")
    top3 = sum(v for _, v in q_prefix.most_common(3)) / n
    for k, v in q_prefix.most_common(top_n):
        print(f"  {v/n:6.1%}  {k!r}")
    print(f"  前 3 前缀合计 = {top3:.1%}" + ("  <-- 塌缩!问题多样性不足[红线]" if top3 > 0.60 else ""))
    if top3 > 0.60:
        red = True

    # ── 答案侧:全部跑在完整答案文本上 ──
    # [3] 答案开头塌缩  [红线: 单一开头 > threshold]
    print(f"\n[3] 答案开头(前12字) Top{top_n}  [红线: 单一开头 > {threshold:.0%}]")
    red |= _print_open(ans_open, n, top_n, threshold)

    # [4] 答案长度 + 啰嗦化告警(答案变长篇报告是头号要避免项)
    print(f"\n[4] 答案长度(字符)  [告警: p90 > {LONG_WARN_P90}]")
    p90 = sorted(all_len)[min(n - 1, int(n * 0.9))]
    print(f"  总体: 平均 {statistics.mean(all_len):5.1f}  中位 {statistics.median(all_len):.0f}"
          f"  p90 {p90}  最长 {max(all_len)}"
          + ("  <-- p90 偏长,可能报告化(告警)" if p90 > LONG_WARN_P90 else ""))
    for qt, ls in sorted(len_by_type.items()):
        print(f"  {qt:16s} n={len(ls):5d}  平均 {statistics.mean(ls):5.1f}"
              f"  中位 {statistics.median(ls):5.0f}  最长 {max(ls):5d}")

    # [5] 拒答类正确率(在完整答案上判不确定表达)  [告警: < 90%]
    print("\n[5] 拒答类(unanswerable+ambiguous)  [告警: 正确给出不确定表达 < 90%]")
    share = refusal_total / n
    band = "" if 0.05 <= share <= 0.10 else "  (不在 5%-10% 目标带)"
    print(f"  占比: {refusal_total}/{n} = {share:.1%}{band}")
    if refusal_total:
        rate = refusal_ok / refusal_total
        print(f"  正确给出不确定/多解表达: {refusal_ok}/{refusal_total} = {rate:.0%}"
              + ("  <-- 偏低,部分拒答题被强答(告警)" if rate < 0.9 else ""))
    else:
        print("  拒答类: 0 条(建议构造 5%-10%)")

    # [6] 残留格式标签自检(纯蒸馏应为 0)  [告警]
    print(f"\n[6] 残留 <analysis>/<answer> 标签: {tag_residual} 条"
          + ("  <-- 蒸馏 prompt 可能漏网(告警)" if tag_residual else "  (干净)"))

    # [7] 越界/高风险词扫描(只读,替代已撤掉的格式校验)  [告警]
    print(f"\n[7] 越界/高风险词命中: {risk_rows} 条 ({risk_rows/n:.1%})"
          + ("  <-- 建议人工抽看这些(告警)" if risk_rows else "  (无)"))
    for w, c in risk_words.most_common(top_n):
        print(f"  {c:5d}  {w}")

    # [8] 编码自检(乱码)
    cyr_rate = cyr / total_chars if total_chars else 0
    print(f"\n[8] 编码自检: 中文 {cjk/total_chars:.0%}  西里尔(乱码嫌疑) {cyr_rate:.2%}"
          + ("  <-- 注意可能乱码" if cyr_rate > 0.005 else "  (正常)"))

    print(f"\n{'='*64}")
    print("结论: " + ("[!] 触发红线,建议处理后再发布"
                      if red else "[OK] 未触发红线(告警项见上,纯蒸馏靠人工抽检兜底)"))
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
