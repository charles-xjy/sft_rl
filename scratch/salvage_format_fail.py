"""
回填脚本:对已落盘的 SFT 数据应用容错补标签(salvage_answer),救回因漏
收尾标签(常见漏 </answer>)而被判 format_fail 的样本,并重新跑自动质检。

用法:
    python scratch/salvage_format_fail.py outputs/03_answers_sft.jsonl

默认原地改写,改写前自动备份为 <file>.bak。--dry-run 只统计不写盘。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validator import salvage_answer, validate_answer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="回填:对历史数据补缺失收尾标签")
    parser.add_argument("data", help="SFT jsonl(如 outputs/03_answers_sft.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="只统计,不写盘")
    args = parser.parse_args()

    path = Path(args.data)
    records = [json.loads(line) for line in path.open(encoding="utf-8")]

    salvaged = recovered = 0  # 补了标签的数 / 由 format_fail 转为 pass 的数
    for r in records:
        meta = r.get("meta", {})
        av = meta.get("auto_validation", {})
        # 只处理之前未通过的(format_fail/warning);pass 的不动。
        if av.get("label") == "pass":
            av.setdefault("salvaged", False)
            continue

        msg = r["messages"][1]
        mode = meta.get("answer_mode", "analysis")
        expect_analysis = mode == "analysis"
        new_content, did = salvage_answer(msg["content"], expect_analysis=expect_analysis)
        if did:
            salvaged += 1
            msg["content"] = new_content
            is_valid, warnings, label = validate_answer(
                new_content, meta.get("question_type", ""), expect_analysis=expect_analysis
            )
            if av.get("label") != "pass" and label == "pass":
                recovered += 1
            av.update({"is_valid": is_valid, "warnings": warnings, "label": label})
        av["salvaged"] = did

    print(f"总样本 {len(records)}  补标签 {salvaged}  format_fail→pass {recovered}")

    if args.dry_run:
        print("[dry-run] 未写盘")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已写回 {path}（备份 {backup}）")


if __name__ == "__main__":
    main()
