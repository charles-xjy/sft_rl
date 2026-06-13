"""
Dataset scanning and sampling helpers.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


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
    """Scan a dataset and return matching image paths."""
    config = DATASET_CONFIGS[dataset_name]
    images: List[Path] = []

    for pattern in config["glob_patterns"]:
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            pattern_with_ext = pattern.replace(".*", ext)
            images.extend(dataset_root.glob(pattern_with_ext))

    images = [p for p in images if config["image_filter"](p)]
    return sorted(set(images))


def scan_all_datasets(dataset_root: Path, datasets: List[str]) -> Dict[str, List[Path]]:
    """Scan multiple datasets and return a dataset-to-images mapping."""
    result: Dict[str, List[Path]] = {}
    for dataset_name in datasets:
        if dataset_name in DATASET_CONFIGS:
            result[dataset_name] = scan_dataset(dataset_root / dataset_name, dataset_name)
    return result


def stratified_sample(
    images_by_dataset: Dict[str, List[Path]],
    num_samples: int,
    seed: int = 42,
) -> List[Tuple[str, Path]]:
    """Randomly sample from the union of all datasets."""
    rng = random.Random(seed)
    all_samples: List[Tuple[str, Path]] = []

    for dataset, images in images_by_dataset.items():
        shuffled = list(images)
        rng.shuffle(shuffled)
        all_samples.extend((dataset, img) for img in shuffled)

    rng.shuffle(all_samples)
    return all_samples[:num_samples]


def sample_per_dataset(
    images_by_dataset: Dict[str, List[Path]],
    samples_per_dataset: int,
    seed: int = 42,
) -> List[Tuple[str, Path]]:
    """Sample the same number of images from each dataset."""
    rng = random.Random(seed)
    samples: List[Tuple[str, Path]] = []

    for dataset, images in images_by_dataset.items():
        shuffled = list(images)
        rng.shuffle(shuffled)
        selected = shuffled[:samples_per_dataset]
        samples.extend((dataset, img) for img in selected)

    return samples


def load_processed_records(output_file: Path, key_field: str = "image_path") -> set:
    """Load processed records from an output file for resume support."""
    processed = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if key_field in record:
                        processed.add(record[key_field])
                except Exception:
                    pass
    return processed
