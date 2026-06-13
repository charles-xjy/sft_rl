"""
自动质检规则模块
"""
import re
from typing import List, Tuple

# 高风险词汇列表
HIGH_RISK_WORDS = [
    "坐标", "经度", "纬度", "GPS", "GNSS", "InSAR",
    "面积", "平方米", "平方公里", "公顷",
    "米", "公里", "千米", "位移", "距离",
    "成因", "原因", "导致", "引发",
    "建议", "应该", "需要", "必须", "方案", "治理", "修复", "避让",
    "禁建区", "缓冲区", "监测", "传感器",
    "政府", "部门", "赔偿", "补偿", "安置",
]

# 问题类型配置
QUESTION_TYPES = [
    ("existence", "是否存在某类地物"),
    ("attribute", "颜色、形状、屋顶类型、材质或纹理"),
    ("location", "某地物位于图像哪个区域"),
    ("spatial_relation", "两个地物之间的相对位置或邻接关系"),
    ("counting", "清晰可见目标的数量或估计数量"),
    ("scene", "整体场景类型或主要功能区"),
    ("comparison", "不同区域的建筑、植被、水体或道路密度对比"),
    ("reasoning", "基于可见证据的简单判断"),
]


def validate_answer(answer: str, question_type: str) -> Tuple[bool, List[str], str]:
    """
    自动质检答案

    Returns:
        (是否通过, 警告列表, 自动标签)
    """
    warnings = []

    # 检查是否同时包含 analysis 和 answer 标签
    if "<analysis>" not in answer or "</analysis>" not in answer:
        return False, ["缺少 analysis 标签"], "format_fail"
    if "<answer>" not in answer or "</answer>" not in answer:
        return False, ["缺少 answer 标签"], "format_fail"

    # 提取 analysis 内容，检查步骤数
    analysis_match = re.search(r"<analysis>(.*?)</analysis>", answer, re.DOTALL)
    if not analysis_match:
        return False, ["无法解析 analysis 内容"], "format_fail"

    analysis_content = analysis_match.group(1).strip()
    steps = [s for s in analysis_content.split("\n") if re.match(r"^\d+\.", s.strip())]

    if len(steps) < 2 or len(steps) > 4:
        warnings.append(f"analysis 步骤数为 {len(steps)}，建议 2-4 步")

    # 检查高风险词汇
    for word in HIGH_RISK_WORDS:
        if word in answer:
            warnings.append(f"包含高风险词汇: {word}")

    # 按问题类型做专项检查
    answer_match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1)

        if question_type == "counting":
            # 计数题需要有数字或不确定表达
            if not re.search(r"\d+|约|估计|不确定|无法确认", answer_text):
                warnings.append("计数题答案中没有数量或估计表达")

        if question_type in ["location", "spatial_relation"]:
            # 位置题需要有方位表达
            location_words = ["左", "右", "上", "下", "中", "侧", "部", "中央", "相邻", "附近", "之间", "旁"]
            if not any(w in answer_text for w in location_words):
                warnings.append("位置/空间关系题答案中没有方位或关系表达")

    # 判断是否通过
    has_high_risk = any("高风险词汇" in w for w in warnings)
    is_pass = not has_high_risk
    auto_label = "pass" if is_pass else "warning"

    return is_pass, warnings, auto_label
