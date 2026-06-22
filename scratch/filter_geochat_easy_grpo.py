#!/usr/bin/env python3
"""Extract GeoChat_Instruct samples whose answers are easy to verify automatically."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCENE_LABEL_ZH = {
    "airplane": "飞机",
    "airport": "机场",
    "baseball diamond": "棒球场",
    "basketball court": "篮球场",
    "beach": "海滩",
    "bridge": "桥梁",
    "chaparral": "灌木丛",
    "church": "教堂",
    "circular farmland": "环形农田",
    "cloud": "云层",
    "commercial area": "商业区",
    "dense residential": "密集居民区",
    "desert": "沙漠",
    "forest": "森林",
    "freeway": "高速公路",
    "golf course": "高尔夫球场",
    "ground track field": "田径场",
    "harbor": "港口",
    "industrial area": "工业区",
    "intersection": "交叉路口",
    "island": "岛屿",
    "lake": "湖泊",
    "meadow": "草地",
    "medium residential": "中密度居民区",
    "mobile home park": "活动房屋区",
    "mountain": "山地",
    "overpass": "立交桥",
    "palace": "宫殿",
    "parking lot": "停车场",
    "railway": "铁路",
    "railway station": "火车站",
    "rectangular farmland": "矩形农田",
    "river": "河流",
    "roundabout": "环岛",
    "runway": "跑道",
    "sea ice": "海冰",
    "ship": "船只",
    "snowberg": "冰雪地",
    "sparse residential": "稀疏居民区",
    "stadium": "体育场",
    "storage tank": "储罐区",
    "tennis court": "网球场",
    "terrace": "梯田",
    "thermal power station": "火力发电站",
    "wetland": "湿地",
}
NOUN_ZH = {
    "road": "道路",
    "roads": "道路",
    "water area": "水域",
    "water areas": "水域",
    "commercial building": "商业建筑",
    "commercial buildings": "商业建筑",
    "residential building": "居民建筑",
    "residential buildings": "居民建筑",
    "building": "建筑物",
    "buildings": "建筑物",
    "grass area": "草地区域",
    "grass areas": "草地区域",
    "forest": "森林",
    "forests": "森林",
    "farmland": "农田",
    "farmlands": "农田",
    "wetland": "湿地",
    "wetlands": "湿地",
    "island": "岛屿",
    "islands": "岛屿",
    "orchard": "果园",
    "orchards": "果园",
    "scrub": "灌木地",
    "scrubs": "灌木地",
    "meadow": "草地",
    "meadows": "草地",
    "cemetery": "墓地",
    "cemeterys": "墓地",
    "industrial": "工业设施",
    "industrials": "工业设施",
    "wood": "林地",
    "woods": "林地",
    "nature reserve": "自然保护区",
    "nature reserves": "自然保护区",
    "heath": "荒原",
    "heaths": "荒原",
    "golf course": "高尔夫球场",
    "golf courses": "高尔夫球场",
    "place of worship": "宗教建筑",
    "place of worships": "宗教建筑",
    "parking": "停车场",
    "parkings": "停车场",
    "pitch": "场地",
    "pitchs": "场地",
    "village green": "村庄绿地",
    "village greens": "村庄绿地",
    "garden": "花园",
    "gardens": "花园",
    "pier": "码头",
    "piers": "码头",
    "playground": "操场",
    "playgrounds": "操场",
    "school": "学校",
    "schools": "学校",
    "residential area": "居民区",
    "residential areas": "居民区",
    "commercial area": "商业区",
    "commercial areas": "商业区",
}
ADJ_ZH = {
    "small": "小型",
    "medium": "中型",
    "large": "大型",
    "rectangular": "矩形",
    "circular": "环形",
    "square": "方形",
}
REL_ZH = {
    "on the left of": "左侧",
    "on the right of": "右侧",
    "at the top of": "上方",
    "at the bottom of": "下方",
    "next to": "旁边",
}
SCENE_CHOICES_PER_SAMPLE = 6
SCENE_RE = re.compile(
    r"classify the given image.*?classes:\s*(.*?)\.?\s*answer in one word or a short phrase\.?$",
    re.IGNORECASE,
)
RURAL_URBAN_RE = re.compile(
    r"^is it a rural or an urban area\??\s*answer in one word or a short phrase\.?$",
    re.IGNORECASE,
)
FLOOD_BINARY_RE = re.compile(
    r"^what is the overall condition of the given image\? flooded or non flooded\.?\s*answer in one word or a short phrase\.?$",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def question_of(item: dict) -> str:
    conversations = item.get("conversations") or []
    if not conversations:
        return ""
    return normalize_text(conversations[0].get("value", "").replace("<image>", ""))


def answer_of(item: dict) -> str:
    conversations = item.get("conversations") or []
    if len(conversations) < 2:
        return ""
    return normalize_text(conversations[1].get("value", ""))


def yesno_subtype(question: str) -> str:
    ql = question.lower()
    if re.match(r"^is the number of .* equal to the number of ", ql):
        return "count_equal_yesno"
    if re.match(r"^(are there|is there|is a |is an |are any )", ql):
        if re.search(r"\bmore\b|\bless\b|\bfewer\b|\blarger\b|\bsmaller\b|\bthan\b", ql):
            return "comparison_yesno"
        return "presence_yesno"
    if re.match(r"^(is|are|does|do|can) ", ql):
        if re.search(r"\bmore\b|\bless\b|\bfewer\b|\blarger\b|\bsmaller\b|\bthan\b", ql):
            return "comparison_yesno"
    return "other_yesno"


def build_row(item: dict, question: str, reference: str, question_type: str) -> dict:
    image_rel = item.get("image") or item.get("id")
    return {
        "_question_type": question_type,
        "image": image_rel,
        "question": question,
        "reference": reference,
    }


def classify_verifiable(item: dict) -> dict | None:
    question = question_of(item)
    reference = answer_of(item)
    if not question or not reference:
        return None

    ref_lower = reference.lower()

    scene_match = SCENE_RE.match(question)
    if scene_match:
        classes = [normalize_text(x).lower() for x in scene_match.group(1).split(",")]
        classes = [x for x in classes if x]
        folder_label = ""
        image_rel = item.get("image") or item.get("id")
        if image_rel:
            parts = Path(image_rel).parts
            if len(parts) >= 2:
                folder_label = parts[-2].replace("_", " ").lower()
        if ref_lower == "none":
            return None
        if ref_lower not in classes:
            return None
        if folder_label and ref_lower != folder_label:
            return None

        row = build_row(item, question, reference, "scene_classification")
        row["_scene_candidates"] = classes
        return row

    if RURAL_URBAN_RE.match(question) and ref_lower in {"rural", "urban"}:
        return build_row(item, question, reference, "rural_urban")

    if FLOOD_BINARY_RE.match(question) and ref_lower in {"flooded", "non flooded", "non-flooded"}:
        return build_row(item, question, reference, "flood_binary")

    if ref_lower in {"yes", "no", "yes.", "no."}:
        return build_row(item, question, reference, "binary_yesno")

    return None


def balanced_sample(rows: list[dict], label_key: str, target: int, seed: int) -> list[dict]:
    if target <= 0 or len(rows) <= target:
        return rows[:]

    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[label_key])].append(row)

    labels = sorted(groups)
    for label in labels:
        rng.shuffle(groups[label])

    selected: list[dict] = []
    used = {label: 0 for label in labels}
    base = target // len(labels)

    for label in labels:
        take = min(base, len(groups[label]))
        selected.extend(groups[label][:take])
        used[label] = take

    remaining = target - len(selected)
    while remaining > 0:
        progressed = False
        for label in labels:
            if remaining == 0:
                break
            if used[label] < len(groups[label]):
                selected.append(groups[label][used[label]])
                used[label] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return selected


def first_n_per_group(rows: list[dict], group_key: str, n: int) -> list[dict]:
    if n <= 0:
        return rows[:]
    counts: Counter[str] = Counter()
    selected: list[dict] = []
    for row in rows:
        group = str(row[group_key])
        if counts[group] >= n:
            continue
        selected.append(row)
        counts[group] += 1
    return selected


def first_n_per_subtype_label(
    rows: list[dict], subtype_key: str, label_key: str, n_per_subtype: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    by_subtype: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subtype[str(row[subtype_key])].append(row)

    selected: list[dict] = []
    for subtype in sorted(by_subtype):
        subtype_rows = by_subtype[subtype][:]
        rng.shuffle(subtype_rows)
        selected.extend(
            balanced_sample(
                subtype_rows,
                label_key=label_key,
                target=n_per_subtype,
                seed=seed,
            )
        )
    return selected


def exclude_rows_by_value(rows: list[dict], key: str, excluded_values: set[str]) -> list[dict]:
    return [row for row in rows if str(row[key]) not in excluded_values]


def scene_folder_label(row: dict) -> str:
    image_rel = row["image"]
    parts = Path(image_rel).parts
    return parts[-2].replace("_", " ").lower()


def answer_label(row: dict) -> str:
    ref = row["reference"].strip().lower()
    if ref in {"yes", "yes."}:
        return "yes"
    if ref in {"no", "no."}:
        return "no"
    if ref in {"non flooded", "non-flooded"}:
        return "non flooded"
    return ref


def row_yesno_subtype(row: dict) -> str:
    return yesno_subtype(row["question"])


def sample_scene_candidates(row: dict, limit: int = SCENE_CHOICES_PER_SAMPLE) -> list[str]:
    candidates = list(row.get("_scene_candidates", []))
    answer = row["reference"].strip().lower()
    if len(candidates) <= limit:
        return candidates

    others = [x for x in candidates if x != answer]
    seed_text = f"{row['image']}|{answer}"
    seed = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(others)
    picked = others[: max(limit - 1, 0)] + [answer]
    rng.shuffle(picked)
    return picked


def translate_phrase(phrase: str) -> str:
    text = normalize_text(phrase.lower())
    text = re.sub(r"\bin the image\b", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(a|an|the)\s+", "", text)

    for rel_en, rel_zh in REL_ZH.items():
        marker = f" {rel_en} "
        if marker in text:
            left, right = text.split(marker, 1)
            return f"{translate_phrase(right)}{rel_zh}的{translate_phrase(left)}"

    if text in NOUN_ZH:
        return NOUN_ZH[text]

    tokens = text.split()
    prefixes = []
    while tokens and tokens[0] in ADJ_ZH:
        prefixes.append(ADJ_ZH[tokens.pop(0)])
    core = " ".join(tokens).strip()
    core_zh = NOUN_ZH.get(core, core)
    return "".join(prefixes) + core_zh


def zh_reference(row: dict) -> str:
    ref = row["reference"].strip().lower()
    question_type = row["_question_type"]
    if question_type == "scene_classification":
        return SCENE_LABEL_ZH.get(ref, ref)
    if question_type == "binary_yesno":
        return "是" if ref in {"yes", "yes."} else "否"
    if question_type == "rural_urban":
        return "农村区域" if ref == "rural" else "城市区域"
    if question_type == "flood_binary":
        return "非洪涝" if ref in {"non flooded", "non-flooded"} else "洪涝"
    return row["reference"]


def zh_question(row: dict) -> str:
    question = row["question"]
    question_type = row["_question_type"]

    if question_type == "scene_classification":
        scene_candidates = sample_scene_candidates(row)
        labels = "、".join(
            SCENE_LABEL_ZH.get(label, label) for label in scene_candidates
        )
        return (
            "请判断这张遥感图像的主要场景类别。"
            f"可选类别：{labels}。"
            "请只输出一个类别名称。"
        )

    if question_type == "rural_urban":
        return "这张遥感图像展示的是农村区域还是城市区域？请用简短词语回答。"

    if question_type == "flood_binary":
        return "这张遥感图像整体属于洪涝还是非洪涝状态？请用简短词语回答。"

    q = normalize_text(question)
    m = re.match(
        r"^Is the number of (.*?) equal to the number of (.*?)\? Answer in one word or a short phrase\.$",
        q,
        re.IGNORECASE,
    )
    if m:
        return (
            f"图中{translate_phrase(m.group(1))}的数量和"
            f"{translate_phrase(m.group(2))}的数量一样多吗？请只回答是或否。"
        )

    m = re.match(
        r"^(Are there|Is there) (more|less|fewer) (.*?) than (.*?)\? Answer in one word or a short phrase\.$",
        q,
        re.IGNORECASE,
    )
    if m:
        cmp_zh = "更多" if m.group(2).lower() == "more" else "更少"
        return (
            f"图中{translate_phrase(m.group(3))}比{translate_phrase(m.group(4))}"
            f"{cmp_zh}吗？请只回答是或否。"
        )

    m = re.match(
        r"^Is the entire (.*?) (flooded|non flooded)\? Answer in one word or a short phrase\.$",
        q,
        re.IGNORECASE,
    )
    if m:
        state_zh = "都被淹没" if m.group(2).lower() == "flooded" else "都处于非洪涝状态"
        return f"图中的整片{translate_phrase(m.group(1))}{state_zh}吗？请只回答是或否。"

    m = re.match(
        r"^(Is there|Is a|Is an)\s+(.*?)(?:\s+present)?(?:\s+in the image)?\? Answer in one word or a short phrase\.$",
        q,
        re.IGNORECASE,
    )
    if m:
        return f"图中有{translate_phrase(m.group(2))}吗？请只回答是或否。"

    return "请根据这张遥感图像回答问题，并只给出简短答案。"


def localize_row(row: dict) -> dict:
    return {
        "image": row["image"],
        "question": zh_question(row),
        "reference": zh_reference(row),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean_row = localize_row(row)
            f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")


def print_report(rows_by_type: dict[str, list[dict]]) -> None:
    for question_type, rows in sorted(rows_by_type.items()):
        print(f"\n[{question_type}] total={len(rows)}")
        if question_type == "scene_classification":
            label_counts = Counter(row["reference"].lower() for row in rows)
            print(f"labels={len(label_counts)}")
            for label, count in sorted(label_counts.items()):
                print(f"  {label}: {count}")
        elif question_type in {"binary_yesno", "rural_urban", "flood_binary"}:
            label_counts = Counter(answer_label(row) for row in rows)
            print("labels:")
            for label, count in sorted(label_counts.items()):
                print(f"  {label}: {count}")
            if question_type == "binary_yesno":
                subtype_counts = Counter(row_yesno_subtype(row) for row in rows)
                print("subtypes:")
                for subtype, count in subtype_counts.most_common():
                    print(f"  {subtype}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(SCRIPT_DIR / "GeoChat_Instruct.json"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "data"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--sample-rural-urban", type=int, default=500)
    parser.add_argument("--sample-flood-binary", type=int, default=500)
    parser.add_argument("--scene-per-folder", type=int, default=30)
    parser.add_argument("--yesno-per-subtype", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows_by_type: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        row = classify_verifiable(item)
        if row is None:
            continue
        rows_by_type[row["_question_type"]].append(row)

    print(f"input: {input_path}")
    print_report(rows_by_type)

    write_jsonl(output_dir / "scene_classification.jsonl", rows_by_type["scene_classification"])
    write_jsonl(output_dir / "binary_yesno.jsonl", rows_by_type["binary_yesno"])
    write_jsonl(output_dir / "rural_urban.jsonl", rows_by_type["rural_urban"])
    write_jsonl(output_dir / "flood_binary.jsonl", rows_by_type["flood_binary"])

    rural_sample = balanced_sample(
        rows_by_type["rural_urban"],
        label_key="reference",
        target=args.sample_rural_urban,
        seed=args.seed,
    )
    flood_sample = balanced_sample(
        rows_by_type["flood_binary"],
        label_key="reference",
        target=args.sample_flood_binary,
        seed=args.seed,
    )
    write_jsonl(output_dir / f"rural_urban_{len(rural_sample)}.jsonl", rural_sample)
    write_jsonl(output_dir / f"flood_binary_{len(flood_sample)}.jsonl", flood_sample)

    scene_first30 = first_n_per_group(
        [
            {**row, "_folder_label": scene_folder_label(row)}
            for row in rows_by_type["scene_classification"]
        ],
        group_key="_folder_label",
        n=args.scene_per_folder,
    )
    scene_first30 = [{k: v for k, v in row.items() if k != "_folder_label"} for row in scene_first30]
    yesno_4x500 = first_n_per_subtype_label(
        [
            {**row, "_yesno_subtype": row_yesno_subtype(row), "_label": answer_label(row)}
            for row in rows_by_type["binary_yesno"]
        ],
        subtype_key="_yesno_subtype",
        label_key="_label",
        n_per_subtype=args.yesno_per_subtype,
        seed=args.seed,
    )
    yesno_4x500 = exclude_rows_by_value(yesno_4x500, "_yesno_subtype", {"other_yesno"})
    yesno_4x500 = [
        {k: v for k, v in row.items() if k not in {"_yesno_subtype", "_label"}} for row in yesno_4x500
    ]
    write_jsonl(
        output_dir / f"scene_classification_first{args.scene_per_folder}_per_folder.jsonl",
        scene_first30,
    )
    write_jsonl(
        output_dir / f"binary_yesno_{args.yesno_per_subtype}_per_subtype.jsonl",
        yesno_4x500,
    )

    print("\n[samples]")
    print(f"rural_urban_sample={len(rural_sample)}")
    print(f"flood_binary_sample={len(flood_sample)}")
    print(f"scene_first{args.scene_per_folder}_per_folder={len(scene_first30)}")
    print(f"binary_yesno_{args.yesno_per_subtype}_per_subtype={len(yesno_4x500)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
