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


def parse_per_dataset_spec(spec_str: str) -> Dict[str, int]:
    """Parse a per-dataset sample spec like "EBD=140,LEVIR-CD+=200,SECOND=200".

    Returns a dict mapping dataset name to sample count.
    A single integer string ("10") is rejected here — the caller should detect
    that case and fall back to the uniform-count code path.
    """
    spec: Dict[str, int] = {}
    for chunk in spec_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"invalid per-dataset spec entry: {chunk!r} "
                f"(expected 'NAME=COUNT' like 'LEVIR-CD+=200')"
            )
        name, _, count_str = chunk.partition("=")
        name = name.strip()
        count_str = count_str.strip()
        try:
            count = int(count_str)
        except ValueError:
            raise ValueError(f"invalid count for dataset {name!r}: {count_str!r}")
        if count < 0:
            raise ValueError(f"sample count must be non-negative: {name}={count}")
        spec[name] = count
    return spec


def sample_ebd_per_disaster(
    images: List[Path],
    per_disaster: int,
    seed: int = 42,
) -> List[Path]:
    """Sample N post-disaster images from each EBD disaster sub-directory.

    EBD layout: dataset/EBD/<DISASTER_NAME>/images/*_post_disaster.*
    The disaster name is recovered from `image.parent.parent.name`.
    """
    rng = random.Random(seed)
    by_disaster: Dict[str, List[Path]] = {}
    for img in images:
        try:
            disaster = img.parent.parent.name  # .../<DISASTER>/images/<file>
        except Exception:
            disaster = "_unknown_"
        by_disaster.setdefault(disaster, []).append(img)

    selected: List[Path] = []
    for disaster in sorted(by_disaster.keys()):
        bucket = list(by_disaster[disaster])
        rng.shuffle(bucket)
        selected.extend(bucket[:per_disaster])
    return selected


def sample_with_spec(
    images_by_dataset: Dict[str, List[Path]],
    spec: Dict[str, int],
    seed: int = 42,
    ebd_per_disaster: int = None,
) -> List[Tuple[str, Path]]:
    """Sample images per dataset according to `spec`.

    `spec[name]` controls the count for each dataset. A dataset that is not in
    `spec` is skipped entirely (no implicit default). When `ebd_per_disaster`
    is set and EBD is present in `images_by_dataset`, EBD is sampled by
    drawing N images from each disaster sub-directory instead of using
    `spec.get("EBD")`.
    """
    rng = random.Random(seed)
    samples: List[Tuple[str, Path]] = []

    for dataset, images in images_by_dataset.items():
        if dataset == "EBD" and ebd_per_disaster is not None:
            selected = sample_ebd_per_disaster(images, ebd_per_disaster, seed=seed)
            samples.extend(("EBD", img) for img in selected)
            continue

        if dataset not in spec:
            continue
        count = spec[dataset]
        if count <= 0:
            continue
        shuffled = list(images)
        rng.shuffle(shuffled)
        samples.extend((dataset, img) for img in shuffled[:count])

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


def load_manifest(manifest_path: Path) -> List[Tuple[str, Path]]:
    """Load a manifest file of sampled images."""
    samples: List[Tuple[str, Path]] = []
    if not manifest_path.exists():
        return samples

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                dataset = record["dataset"]
                image_path = Path(record["image_path"])
                samples.append((dataset, image_path))
            except Exception:
                pass
    return samples


def save_manifest(manifest_path: Path, samples: List[Tuple[str, Path]]) -> None:
    """Persist sampled images to a manifest file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for dataset, image_path in samples:
            record = {
                "dataset": dataset,
                "image_path": str(Path(image_path).resolve()),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
