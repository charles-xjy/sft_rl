"""
Automatic validation rules for generated answers.
"""
import re
from typing import List, Tuple


# 真正越界单图 VQA 的词：出现即阻断（降级为 warning，不进 pass）。
# 这些表达基本只可能来自图像外推断（坐标/测量/工程治理/行政处置）。
HARD_RISK_WORDS = [
    "坐标",
    "经度",
    "纬度",
    "GPS",
    "GNSS",
    "InSAR",
    "禁建区",
    "缓冲区",
    "避让",
    "治理",
    "修复",
    "方案",
    "传感器",
    "赔偿",
    "补偿",
    "安置",
]


# 日常高频词：在遥感答案里常作正常表达（"需要进一步确认""屋顶颜色导致反光"
# "政府大楼"等），仅作软提示，不阻断 pass，避免误杀。
SOFT_RISK_WORDS = [
    "成因",
    "原因",
    "导致",
    "引发",
    "建议",
    "应该",
    "需要",
    "必须",
    "监测",
    "政府",
    "部门",
]


CONTEXTUAL_RISK_PATTERNS = [
    r"面积\s*(约|为|是)?\s*[0-9零一二三四五六七八九十百千万两\d]+",
    r"[0-9零一二三四五六七八九十百千万两\d]+\s*(平方米|平方公里|公顷)",
    r"距离\s*(约|为|是)?\s*[0-9零一二三四五六七八九十百千万两\d]+",
    r"[0-9零一二三四五六七八九十百千万两\d]+\s*(米|公里|千米)",
]


LOCATION_WORDS = [
    "左上",
    "左下",
    "右上",
    "右下",
    "左侧",
    "右侧",
    "上方",
    "下方",
    "中部",
    "中央",
    "附近",
    "相邻",
    "之间",
]


META_REASONING_PATTERNS = [
    r"我认为",
    r"我猜测",
    r"让我",
    r"先分析",
    r"分析用户请求",
    r"构建最终答案",
    r"格式化输出",
    r"最终审查",
]


def _extract_sections(answer: str):
    analysis_match = re.search(r"<analysis>(.*?)</analysis>", answer, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    analysis_text = analysis_match.group(1).strip() if analysis_match else ""
    answer_text = answer_match.group(1).strip() if answer_match else ""
    return analysis_match, answer_match, analysis_text, answer_text


def _find_direct_risks(text: str) -> List[str]:
    warnings = []
    for word in HARD_RISK_WORDS:
        if word in text:
            warnings.append(f"包含高风险词汇: {word}")
    for word in SOFT_RISK_WORDS:
        if word in text:
            warnings.append(f"包含需复核词汇: {word}")
    for pattern in CONTEXTUAL_RISK_PATTERNS:
        if re.search(pattern, text):
            warnings.append("包含疑似精确数值/单位表达")
    return warnings


def salvage_answer(answer: str, expect_analysis: bool = True) -> Tuple[str, bool]:
    """
    容错补标签。

    模型对超短答案(existence 的"是")倾向答完即发结束符(EOS),漏掉收尾的
    `</answer>`(偶尔 analysis 模式连 `</analysis>` 也一起漏)。这类样本答案内容
    本身完整正确,仅缺收尾标签,会被 validate_answer 误判为 format_fail。此函数把
    缺失的收尾标签补回,救回绝大多数 format_fail。

    只在 `<answer>` 开标签存在时补救;连开标签都没有的属真正格式错误,原样返回。

    Returns:
        (salvaged_answer, was_salvaged)
    """
    if "<answer>" not in answer:
        return answer, False

    text = answer.rstrip()
    salvaged = False

    # analysis 模式:<analysis> 开了没关,且其后直接进入 <answer> —— 在 <answer> 前补 </analysis>。
    if expect_analysis and "<analysis>" in text and "</analysis>" not in text:
        text = text.replace("<answer>", "</analysis>\n<answer>", 1)
        salvaged = True

    # 缺收尾 </answer>:补在末尾。
    if "</answer>" not in text:
        text = text + "</answer>"
        salvaged = True

    return (text, True) if salvaged else (answer, False)


def validate_answer(
    answer: str, question_type: str, expect_analysis: bool = True
) -> Tuple[bool, List[str], str]:
    """
    Validate a generated answer.

    Args:
        expect_analysis: True 为带 analysis 的主力样本；False 为直答(无 analysis)样本。

    Returns:
        (is_valid, warnings, auto_label)
    """
    warnings: List[str] = []

    # <answer> 两种模式都必须有
    if "<answer>" not in answer or "</answer>" not in answer:
        return False, ["缺少 answer 标签"], "format_fail"

    analysis_match, answer_match, analysis_text, answer_text = _extract_sections(answer)
    if not answer_match:
        return False, ["无法解析 answer 内容"], "format_fail"
    if len(answer_text) == 0:
        return False, ["answer 内容为空"], "format_fail"

    if expect_analysis:
        if "<analysis>" not in answer or "</analysis>" not in answer:
            return False, ["缺少 analysis 标签"], "format_fail"
        if not analysis_match:
            return False, ["无法解析 analysis 内容"], "format_fail"

        # analysis 步数现为自适应(1-3 步),只在异常时提示
        steps = [s.strip() for s in analysis_text.split("\n") if re.match(r"^\d+\.", s.strip())]
        if len(steps) == 0:
            warnings.append("analysis 没有分步证据")
        elif len(steps) > 3:
            warnings.append(f"analysis 步骤数为 {len(steps)}，建议不超过 3 步")

        if len(analysis_text) < 20:
            warnings.append("analysis 过短，证据链可能不足")
        if len(analysis_text) > 220:
            warnings.append("analysis 过长，建议压缩为简短证据链")

        for pattern in META_REASONING_PATTERNS:
            if re.search(pattern, analysis_text):
                warnings.append("analysis 含有元推理或过程废话")
                break

        warnings.extend(_find_direct_risks(analysis_text))
    else:
        # 直答模式不应再出现 analysis
        if "<analysis>" in answer:
            warnings.append("直答样本不应包含 analysis 标签")

    if len(answer_text) > 40:
        warnings.append("answer 过长，建议只保留最终结论")

    warnings.extend(_find_direct_risks(answer_text))

    if question_type == "unanswerable":
        if not re.search(r"无法|不确定|难以确认|不能确定", answer_text):
            warnings.append("unanswerable 题答案未给出拒答/不确定表达")

    if question_type == "ambiguous":
        if not re.search(r"可能|也可能|无法准确区分|难以区分|或", answer_text):
            warnings.append("ambiguous 题答案未给出多解/不确定表达")

    if question_type == "counting":
        if not re.search(r"\d+|零|一|二|三|四|五|六|七|八|九|十|约|大约|无法确认|不确定", answer_text):
            warnings.append("计数题答案中没有数量或不确定表达")

    if question_type in ["location", "spatial_relation"]:
        if not any(word in answer_text or word in analysis_text for word in LOCATION_WORDS):
            warnings.append("位置/空间关系题缺少方位或关系表达")

    if question_type == "existence":
        if not re.search(r"有|没有|存在|不存在|可见|不可见", answer_text):
            warnings.append("存在性问题答案不够直接")

    has_blocking_warning = any(
        "高风险词汇" in warning
        or "疑似精确数值" in warning
        or "格式" in warning
        for warning in warnings
    )
    is_valid = not has_blocking_warning
    auto_label = "pass" if is_valid else "warning"
    return is_valid, warnings, auto_label
