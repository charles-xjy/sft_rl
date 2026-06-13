"""
图像扫描与采样模块
"""
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

# 数据集配置
DATASET_CONFIGS = {
    "EBD": {
        "glob_patterns": ["**/images/*_post_disaster.*"],
        "image_filter": lambda p: "post_disaster" in p.name,
    },
    "LEVIR-CD+": {
        "glob_patterns": ["**/B/*.png"],
        "image_filter": lambda p: p.parent.name == "B",
    },
    "SECOND": {
        "glob_patterns": ["**/im2/*.png"],
        "image_filter": lambda p: p.parent.name == "im2",
    },
}


def scan_dataset(dataset_root: Path, dataset_name: str) -> List[Path]:
    """扫描指定数据集的图像"""
    config = DATASET_CONFIGS[dataset_name]
    images = []

    for pattern in config["glob_patterns"]:
        # 匹配不同扩展名
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            pattern_with_ext = pattern.replace(".*", ext)
            images.extend(dataset_root.glob(pattern_with_ext))

    # 应用过滤器并去重
    images = [p for p in images if config["image_filter"](p)]
    return sorted(set(images))


def scan_all_datasets(dataset_root: Path, datasets: List[str]) -> Dict[str, List[Path]]:
    """扫描多个数据集，返回字典"""
    result = {}
    for dataset_name in datasets:
        if dataset_name in DATASET_CONFIGS:
            images = scan_dataset(dataset_root / dataset_name, dataset_name)
            result[dataset_name] = images
    return result


def stratified_sample(images_by_dataset: Dict[str, List[Path]],
                      num_samples: int,
                      seed: int = 42) -> List[Tuple[str, Path]]:
    """分层采样，保证各数据集都有代表性样本"""
    random.seed(seed)
    all_samples = []

    for dataset, images in images_by_dataset.items():
        random.shuffle(images)
        all_samples.extend([(dataset, img) for img in images])

    random.shuffle(all_samples)
    return all_samples[:num_samples]


def load_processed_records(output_file: Path, key_field: str = "image_path") -> set:
    """从已有的输出文件中加载已处理记录，用于断点续传"""
    processed = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if key_field in record:
                        processed.add(record[key_field])
                except:
                    pass
    return processed


# 需要导入json
import json
