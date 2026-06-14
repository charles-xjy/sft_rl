"""问题类型的分布调度:把"配比控制"从事后降采样前移到生成时(软配额)。

为每张图分配一份"建议类型清单"注入 Phase2 prompt;模型对不适合的图可跳过。
单图独立、按 (seed|image) 确定性抽样,断点续跑/并行结果一致;图像数量足够时
聚合分布会收敛到 SCHEDULE_DIST(大数定律),从而让各类型比例"不差太远"。

注意:这是软约束(prompt 里允许跳过),最终精确比例仍由发布期降采样兜底
(scratch/learn/02_template_health.py --rebalance)。
"""

import hashlib
import random
from collections import Counter
from typing import List

# 生成期目标分布(含拒答类 unanswerable + 多解类 ambiguous,合计拒答类 ≈ 6%)
SCHEDULE_DIST = {
    "existence": 0.33,
    "location": 0.23,
    "spatial_relation": 0.19,
    "attribute": 0.14,
    "counting": 0.05,
    "unanswerable": 0.03,
    "ambiguous": 0.03,
}


def assign_question_types(image_path: str, min_q: int, max_q: int, seed: int = 42) -> List[str]:
    """为一张图确定性地分配 k 个建议类型(k 在 [min_q, max_q],按目标分布带放回抽样)。"""
    h = hashlib.md5(f"{seed}|{image_path}".encode("utf-8")).hexdigest()
    rng = random.Random(int(h[:16], 16))
    k = rng.randint(min_q, max_q)
    types = list(SCHEDULE_DIST.keys())
    weights = list(SCHEDULE_DIST.values())
    return rng.choices(types, weights=weights, k=k)


def format_plan(types: List[str]) -> str:
    """把类型清单渲染成 prompt 用的中文配额说明。"""
    counts = Counter(types)
    order = [t for t in SCHEDULE_DIST if t in counts]
    return "\n".join(f"- {t}：{counts[t]} 个" for t in order)
