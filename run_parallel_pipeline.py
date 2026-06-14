#!/usr/bin/env python3
"""
Run the three-stage pipeline with staged parallelism.

Phase 1 emits descriptions.
Phase 2 starts as soon as a description is ready.
Phase 3 waits until Phase 1 and Phase 2 are both fully complete,
then runs over all pending questions.
Overall concurrent API requests are capped by --max-concurrency.
"""
import argparse
import hashlib
import json
import threading
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Tuple

from tqdm import tqdm

from src.prompts import (
    ANSWER_DIRECT_PROMPT,
    ANSWER_GENERATION_PROMPT,
    DESCRIPTION_PROMPT,
    QUESTION_GENERATION_PROMPT,
)
from src.sampling import assign_question_types, format_plan
from src.scanner import (
    load_manifest,
    parse_per_dataset_spec,
    sample_per_dataset,
    sample_with_spec,
    save_manifest,
    scan_all_datasets,
    stratified_sample,
)
from src.validator import validate_answer
from src.vlm_client import VLMClient


VALID_QUESTION_TYPES = {
    "existence",
    "attribute",
    "location",
    "spatial_relation",
    "counting",
    "unanswerable",
    "ambiguous",
}


def pick_answer_mode(image_path: str, question: str, qtype: str,
                     direct_ratio: float, seed: int) -> str:
    """决定一条问答用 analysis 还是 direct 模式。

    并行环境下不共享 rng,改用 (image|question) 的稳定哈希做确定性抽样:
    同一条样本无论重跑还是断点续跑都得到相同模式,且天然线程安全。
    unanswerable / spatial_relation 需要证据链,固定走 analysis。
    """
    if qtype in ("unanswerable", "ambiguous", "spatial_relation"):
        return "analysis"
    h = hashlib.md5(f"{seed}|{image_path}|{question}".encode("utf-8")).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "direct" if frac < direct_ratio else "analysis"


def parse_questions_response(response: str) -> List[Dict[str, str]]:
    """Parse `type|question` lines from model output."""
    questions: List[Dict[str, str]] = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        q_type, q_content = line.split("|", 1)
        q_type = q_type.strip().lower()
        q_content = q_content.strip()
        if q_type in VALID_QUESTION_TYPES and q_content:
            questions.append({"question_type": q_type, "question": q_content})
    return questions


def append_jsonl(path: Path, record: dict, lock: threading.Lock) -> None:
    """Append one JSON object to a JSONL file."""
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_by_key(path: Path, key_field: str) -> Dict[str, dict]:
    """Load a JSONL file into a dict keyed by one field."""
    records: Dict[str, dict] = {}
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                key = record.get(key_field)
                if key:
                    records[key] = record
            except Exception:
                pass
    return records


def load_question_records(path: Path) -> Dict[str, dict]:
    """Load Phase 2 records keyed by image path."""
    return load_jsonl_by_key(path, "image_path")


def load_answer_keys(path: Path) -> set:
    """Load processed answer keys of the form abs_image_path|question."""
    processed = set()
    if not path.exists():
        return processed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                user_msg = record["messages"][0]["content"]
                question = user_msg.replace("<image>", "").strip()
                key = f"{record['images'][0]}|{question}"
                processed.add(key)
            except Exception:
                pass
    return processed


def make_client_factory(base_url: str, model: str, retries: int):
    """Build a thread-local VLM client factory."""
    local_state = threading.local()

    def get_client() -> VLMClient:
        client = getattr(local_state, "client", None)
        if client is None:
            local_state.client = VLMClient(base_url=base_url, model=model, retries=retries)
            client = local_state.client
        return client

    return get_client


def phase1_task(task: dict, get_client) -> dict:
    """Run Phase 1 for one image."""
    client = get_client()
    image_path = Path(task["image_path"])
    description = client.call(
        prompt=DESCRIPTION_PROMPT,
        image_path=image_path,
        max_tokens=2048,
        temperature=0.3,
    )
    return {
        "kind": "phase1",
        "dataset": task["dataset"],
        "image_path": str(image_path.resolve()),
        "description": description,
    }


def phase2_task(task: dict, get_client, num_questions: int, min_questions: int,
                seed: int = 42) -> dict:
    """Run Phase 2 for one image description."""
    client = get_client()
    image_path = Path(task["image_path"])
    assigned = assign_question_types(task["image_path"], min_questions, num_questions, seed)
    prompt = QUESTION_GENERATION_PROMPT.format(
        assigned_plan=format_plan(assigned),
        description=task["description"],
    )
    response = client.call(
        prompt=prompt,
        image_path=image_path,
        max_tokens=2048,
        temperature=0.4,
    )
    questions = parse_questions_response(response)
    if len(questions) > num_questions:
        questions = questions[:num_questions]
    return {
        "kind": "phase2",
        "dataset": task["dataset"],
        "image_path": str(image_path.resolve()),
        "description": task["description"],
        "questions": questions,
    }


def phase3_task(task: dict, get_client, direct_ratio: float = 0.3, seed: int = 42) -> dict:
    """Run Phase 3 for one question."""
    client = get_client()
    image_path = Path(task["image_path"])
    mode = pick_answer_mode(
        task["image_path"], task["question"], task["question_type"], direct_ratio, seed
    )
    template = ANSWER_GENERATION_PROMPT if mode == "analysis" else ANSWER_DIRECT_PROMPT
    prompt = template.format(
        question=task["question"],
        question_type=task["question_type"],
        description=task["description"],
    )
    answer = client.call(
        prompt=prompt,
        image_path=image_path,
        max_tokens=2048,
        temperature=0.3,
    )
    is_valid, warnings, auto_label = validate_answer(
        answer, task["question_type"], expect_analysis=(mode == "analysis")
    )
    record = {
        "messages": [
            {"role": "user", "content": f"<image>\n{task['question']}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [str(image_path.resolve())],
        "meta": {
            "task_type": "vqa_with_analysis" if mode == "analysis" else "vqa_direct",
            "answer_mode": mode,
            "question_type": task["question_type"],
            "source_dataset": task["dataset"],
            "auto_validation": {
                "is_valid": is_valid,
                "warnings": warnings,
                "label": auto_label,
            },
            "generated_at": datetime.now().isoformat(),
        },
    }
    return {
        "kind": "phase3",
        "dataset": task["dataset"],
        "question_type": task["question_type"],
        "question": task["question"],
        "image_path": str(image_path.resolve()),
        "record": record,
        "is_valid": is_valid,
        "auto_label": auto_label,
    }


def print_summary(stats: dict, desc_path: Path, question_path: Path, answer_path: Path) -> None:
    """Print a compact pipeline summary."""
    print("\n" + "=" * 80)
    print("Pipeline Parallel Summary")
    print("=" * 80)
    print(f"phase1_completed: {stats['phase1_completed']}")
    print(f"phase2_completed: {stats['phase2_completed']}")
    print(f"phase3_completed: {stats['phase3_completed']}")
    print(f"phase3_auto_pass: {stats['phase3_auto_pass']}")
    print(f"errors: {stats['errors']}")
    print(f"descriptions_file: {desc_path}")
    print(f"questions_file: {question_path}")
    print(f"answers_file: {answer_path}")
    print("=" * 80)


def build_relevant_answer_keys(processed_answer_keys: set, sampled_image_paths: set) -> set:
    """Filter processed answer keys down to the current sampled images."""
    relevant = set()
    for key in processed_answer_keys:
        image_path, _, question = key.partition("|")
        if image_path in sampled_image_paths and question:
            relevant.add(key)
    return relevant


def main():
    parser = argparse.ArgumentParser(description="Run the 3-stage pipeline with staged parallelism")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="数据集根目录")
    parser.add_argument("--datasets", type=str, default="LEVIR-CD+,SECOND", help="要处理的数据集，逗号分隔")
    parser.add_argument("--num-images", type=int, default=None, help="全局随机采样总数；传入后优先于 --samples-per-dataset")
    parser.add_argument(
        "--samples-per-dataset",
        type=str,
        default="LEVIR-CD+=500,SECOND=500",
        help=(
            "每个数据集采样数量。两种语法："
            "(1) 单一整数 '10' — 每个数据集都抽 10 张；"
            "(2) 按数据集指定 'LEVIR-CD+=500,SECOND=500' — 仅抽指定的"
        ),
    )
    parser.add_argument(
        "--ebd-per-disaster",
        type=int,
        default=None,
        help="EBD 按 7 种灾害子目录各抽 N 张；启用后会覆盖 --samples-per-dataset 里 EBD 的值",
    )
    parser.add_argument("--num-questions", type=int, default=5, help="每张图最多生成问题数量")
    parser.add_argument("--min-questions", type=int, default=2, help="每张图最少生成问题数量")
    parser.add_argument("--max-concurrency", type=int, default=24, help="整体最大并发请求数")
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--base-url", type=str, default="http://10.129.107.145:8001/v1", help="vLLM 服务地址")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--direct-ratio", type=float, default=0.3,
                        help="不带 analysis 的直答样本占比(打散固定起手式,缓解过度模板化)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--manifest", type=str, default=None, help="样本清单文件路径；默认 outputs/sample_manifest.jsonl")
    parser.add_argument("--refresh-manifest", action="store_true", help="忽略已有 manifest，重新采样并覆盖")
    parser.add_argument("--no-health-check", action="store_true", help="跳过生成结束后的自动数据体检")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    desc_path = output_dir / "01_descriptions.jsonl"
    question_path = output_dir / "02_questions.jsonl"
    answer_path = output_dir / "03_answers_sft.jsonl"
    stats_path = output_dir / "03_stats.json"
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "sample_manifest.jsonl"

    dataset_root = Path(args.dataset_root)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    images_by_dataset = scan_all_datasets(dataset_root, datasets)

    print("=" * 80)
    print("Pipeline Parallel Runner")
    print("=" * 80)
    for dataset, images in images_by_dataset.items():
        print(f"[{dataset}] found {len(images)} images")

    if manifest_path.exists() and not args.refresh_manifest:
        samples = load_manifest(manifest_path)
        print(f"loaded manifest: {manifest_path} ({len(samples)} images)")
    else:
        spec_str = args.samples_per_dataset.strip()
        is_uniform = spec_str.isdigit()

        if args.num_images is not None:
            samples = stratified_sample(images_by_dataset, args.num_images, args.seed)
            print(f"global sampled images: {len(samples)}")
        elif is_uniform and args.ebd_per_disaster is None:
            uniform_count = int(spec_str)
            samples = sample_per_dataset(images_by_dataset, uniform_count, args.seed)
            print(f"per-dataset sampled images: {uniform_count} each, total {len(samples)}")
        else:
            if is_uniform:
                # uniform count + EBD per-disaster: build a spec for non-EBD datasets
                uniform_count = int(spec_str)
                spec = {name: uniform_count for name in datasets if name != "EBD"}
            else:
                spec = parse_per_dataset_spec(spec_str)
                unknown = set(spec) - set(datasets)
                if unknown:
                    print(f"[warning] spec mentions datasets not in --datasets: {sorted(unknown)}")
            samples = sample_with_spec(
                images_by_dataset,
                spec=spec,
                seed=args.seed,
                ebd_per_disaster=args.ebd_per_disaster,
            )
            for dataset in images_by_dataset:
                count = sum(1 for d, _ in samples if d == dataset)
                if dataset == "EBD" and args.ebd_per_disaster is not None:
                    print(
                        f"[{dataset}] sampled {count} images "
                        f"(EBD per-disaster={args.ebd_per_disaster})"
                    )
                else:
                    print(f"[{dataset}] sampled {count} images")
            print(f"total sampled images: {len(samples)}")
        save_manifest(manifest_path, samples)
        print(f"saved manifest: {manifest_path}")

    bootstrap_client = VLMClient(base_url=args.base_url, model=args.model, retries=args.retries)
    resolved_model = bootstrap_client.model
    get_client = make_client_factory(bootstrap_client.base_url, resolved_model, args.retries)

    description_records = load_jsonl_by_key(desc_path, "image_path")
    question_records = load_question_records(question_path)
    processed_answer_keys = load_answer_keys(answer_path)

    desc_lock = threading.Lock()
    question_lock = threading.Lock()
    answer_lock = threading.Lock()

    task_queue: Deque[dict] = deque()
    sampled_image_paths = set()
    phase1_done_initial = 0
    phase2_done_initial = 0
    phase3_done_initial = 0
    phase3_pending_initial = 0

    for dataset, image_path in samples:
        abs_image_path = str(Path(image_path).resolve())
        sampled_image_paths.add(abs_image_path)
        desc_record = description_records.get(abs_image_path)
        question_record = question_records.get(abs_image_path)

        if question_record:
            phase1_done_initial += 1
            phase2_done_initial += 1
            for question_item in question_record.get("questions", []):
                question_key = f"{abs_image_path}|{question_item['question']}"
                if question_key not in processed_answer_keys:
                    phase3_pending_initial += 1
                else:
                    phase3_done_initial += 1
        elif desc_record:
            phase1_done_initial += 1
            task_queue.append({
                "stage": "phase2",
                "dataset": desc_record["dataset"],
                "image_path": abs_image_path,
                "description": desc_record["description"],
            })
        else:
            task_queue.append({
                "stage": "phase1",
                "dataset": dataset,
                "image_path": abs_image_path,
            })

    relevant_processed_answer_keys = build_relevant_answer_keys(processed_answer_keys, sampled_image_paths)
    phase3_done_initial = len(relevant_processed_answer_keys)

    print(f"initial scheduled tasks: {len(task_queue)}")
    print(f"max_concurrency: {args.max_concurrency}")
    print(f"resolved_model: {resolved_model}")
    print("resume support: enabled")
    print("execution mode: P1/P2 parallel, P3 after P1/P2 complete")

    stats = defaultdict(int)
    stats["phase1_completed"] = phase1_done_initial
    stats["phase2_completed"] = phase2_done_initial
    stats["phase3_completed"] = phase3_done_initial
    stats["phase2_questions_generated"] = phase3_done_initial + phase3_pending_initial
    phase1_total = len(samples)
    phase2_total = len(samples)
    phase3_total = phase3_done_initial + phase3_pending_initial
    future_to_task: Dict[object, dict] = {}

    def submit_task(executor: ThreadPoolExecutor, task: dict):
        stage = task["stage"]
        if stage == "phase1":
            future = executor.submit(phase1_task, task, get_client)
        elif stage == "phase2":
            future = executor.submit(phase2_task, task, get_client, args.num_questions, args.min_questions, args.seed)
        elif stage == "phase3":
            future = executor.submit(phase3_task, task, get_client, args.direct_ratio, args.seed)
        else:
            raise ValueError(f"unknown task stage: {stage}")
        future_to_task[future] = task

    phase12_complete = False

    def build_phase3_tasks() -> List[dict]:
        """Build all pending Phase 3 tasks from the latest Phase 2 outputs."""
        question_records_latest = load_question_records(question_path)
        phase3_tasks: List[dict] = []
        for abs_image_path in sampled_image_paths:
            question_record = question_records_latest.get(abs_image_path)
            if not question_record:
                continue
            for question_item in question_record.get("questions", []):
                question_key = f"{abs_image_path}|{question_item['question']}"
                if question_key in processed_answer_keys:
                    continue
                phase3_tasks.append({
                    "stage": "phase3",
                    "dataset": question_record["dataset"],
                    "image_path": abs_image_path,
                    "description": question_record["description"],
                    "question_type": question_item["question_type"],
                    "question": question_item["question"],
                })
        return phase3_tasks

    def maybe_enqueue_phase3():
        """Once P1/P2 are finished, enqueue all pending P3 tasks exactly once."""
        nonlocal phase12_complete, phase3_total
        if phase12_complete:
            return
        if stats["phase1_completed"] < phase1_total:
            return
        if stats["phase2_completed"] < phase2_total:
            return
        phase3_tasks = build_phase3_tasks()
        phase3_total = stats["phase3_completed"] + len(phase3_tasks)
        task_queue.extend(phase3_tasks)
        phase12_complete = True

    def drain_queue(executor: ThreadPoolExecutor):
        while task_queue and len(future_to_task) < args.max_concurrency:
            submit_task(executor, task_queue.popleft())

    with tqdm(total=phase1_total, initial=phase1_done_initial, desc="P1 descriptions", position=0) as p1_bar, \
         tqdm(total=phase2_total, initial=phase2_done_initial, desc="P2 questions", position=1) as p2_bar, \
         tqdm(total=phase3_total, initial=phase3_done_initial, desc="P3 answers", position=2) as p3_bar, \
         ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        drain_queue(executor)

        while future_to_task:
            done, _ = wait(list(future_to_task.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                task = future_to_task.pop(future)
                try:
                    result = future.result()
                except Exception as e:
                    stats["errors"] += 1
                    tqdm.write(f"[error] {task['stage']} | {task.get('image_path', '')}: {e}")
                    drain_queue(executor)
                    continue

                kind = result["kind"]
                if kind == "phase1":
                    description = result["description"]
                    if not description:
                        tqdm.write(f"[empty-output] {Path(result['image_path']).name}: model returned empty description")
                    else:
                        desc_record = {
                            "image_path": result["image_path"],
                            "dataset": result["dataset"],
                            "description": description,
                            "generated_at": datetime.now().isoformat(),
                        }
                        append_jsonl(desc_path, desc_record, desc_lock)
                        stats["phase1_completed"] += 1
                        p1_bar.update(1)
                        p1_bar.set_postfix(done=stats["phase1_completed"], queued=len(task_queue), active=len(future_to_task))
                        task_queue.append({
                            "stage": "phase2",
                            "dataset": result["dataset"],
                            "image_path": result["image_path"],
                            "description": description,
                        })

                elif kind == "phase2":
                    questions = result["questions"]
                    if not questions:
                        tqdm.write(f"[empty-output] {Path(result['image_path']).name}: no valid questions parsed")
                    else:
                        q_record = {
                            "image_path": result["image_path"],
                            "dataset": result["dataset"],
                            "description": result["description"],
                            "questions": questions,
                            "generated_at": datetime.now().isoformat(),
                        }
                        append_jsonl(question_path, q_record, question_lock)
                        stats["phase2_completed"] += 1
                        stats["phase2_questions_generated"] += len(questions)
                        p2_bar.update(1)
                        p2_bar.set_postfix(done=stats["phase2_completed"], questions=stats["phase2_questions_generated"], active=len(future_to_task))

                elif kind == "phase3":
                    append_jsonl(answer_path, result["record"], answer_lock)
                    processed_answer_keys.add(f"{result['image_path']}|{result['question']}")
                    stats["phase3_completed"] += 1
                    p3_bar.update(1)
                    p3_bar.set_postfix(done=stats["phase3_completed"], auto_pass=stats["phase3_auto_pass"], active=len(future_to_task))
                    stats[f"dataset_{result['dataset']}"] += 1
                    stats[f"type_{result['question_type']}"] += 1
                    if result["auto_label"] == "pass":
                        stats["phase3_auto_pass"] += 1
                        p3_bar.set_postfix(done=stats["phase3_completed"], auto_pass=stats["phase3_auto_pass"], active=len(future_to_task))

                maybe_enqueue_phase3()
                p3_bar.total = phase3_total
                p3_bar.refresh()
                drain_queue(executor)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, ensure_ascii=False, indent=2)

    print_summary(stats, desc_path, question_path, answer_path)
    print(f"stats_file: {stats_path}")

    if not args.no_health_check and answer_path.exists():
        run_health_check(answer_path)


def run_health_check(answer_path: Path) -> None:
    """生成结束后自动跑一次只读体检(报告 + 红线告警)。不自动降采样。"""
    import subprocess
    import sys

    script = Path(__file__).parent / "scratch" / "learn" / "02_template_health.py"
    if not script.exists():
        print(f"[health-check] 跳过: 未找到 {script}")
        return
    print("\n" + "=" * 80)
    print("自动体检(只读)。降采样请在人工抽检后手动运行:")
    print(f"  python {script} {answer_path} --rebalance --out sft_data/release.jsonl")
    print("=" * 80, flush=True)  # flush 确保提示在子进程报告之前打印
    # 体检触发红线会以退出码 1 返回,这里不让它中断流水线,仅作提示
    rc = subprocess.run([sys.executable, str(script), str(answer_path)]).returncode
    if rc == 1:
        print("\n[health-check] [!] 体检触发红线,请按上方报告处理后再发布(详见报告)。")


if __name__ == "__main__":
    main()
