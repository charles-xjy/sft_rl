"""
本地人工审批服务:对生成的 VQA 样本逐条通过/不通过。

入口脚本 `02_review.py` 解析命令行后调用 `serve(args)`。
前端静态资源在项目根目录的 review/ui/ 下。
"""
import json
import mimetypes
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

# 项目根目录(src/ 的上一级),用于定位 review/ui 静态资源。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_samples(input_path: Path) -> list:
    samples = []
    if not input_path.exists():
        return samples
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except Exception:
                continue
            try:
                question = record["messages"][0]["content"].replace("<image>", "").strip()
                answer = record["messages"][1]["content"]
                image_path = record["images"][0]
            except Exception:
                continue
            sample_id = f"{image_path}|{question}"
            samples.append({
                "id": sample_id,
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "meta": record.get("meta", {}),
            })
    return samples


def load_predictions(path: Path) -> dict:
    """加载模型预测(05_baseline.py 风格: {image, question, prediction})。

    按 `image|question` 建键,与样本 id 对齐,用于在前端对比教师 vs 学生答案。
    文件不存在则返回空 dict(对比功能为可选)。
    """
    preds: dict = {}
    if not path or not path.exists():
        return preds
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                key = f"{r['image']}|{r['question']}"
                preds[key] = r.get("prediction", "")
            except Exception:
                continue
    return preds


def load_reviews(review_path: Path) -> dict:
    reviews: dict = {}
    if not review_path.exists():
        return reviews
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                reviews[record["id"]] = record
            except Exception:
                continue
    return reviews


def persist_reviews(review_path: Path, reviews: dict) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as f:
        for review in reviews.values():
            f.write(json.dumps(review, ensure_ascii=False) + "\n")


def make_handler(samples: list, review_path: Path, static_dir: Path, predictions: dict = None):
    reviews = load_reviews(review_path)
    predictions = predictions or {}

    class ReviewHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path, content_type=None) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            body = path.read_bytes()
            mime_type = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sample_payload(self, sample: dict) -> dict:
            review = reviews.get(sample["id"])
            return {
                **sample,
                "image_url": f"/api/image?path={quote(sample['image_path'])}",
                "review": review,
                "prediction": predictions.get(sample["id"]),   # 学生模型答案(无则 None)
            }

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/items":
                query = parse_qs(parsed.query)
                filter_mode = query.get("filter", ["all"])[0]
                payload = []
                for sample in samples:
                    has_review = sample["id"] in reviews
                    if filter_mode == "pending" and has_review:
                        continue
                    if filter_mode == "approved" and reviews.get(sample["id"], {}).get("decision") != "approved":
                        continue
                    if filter_mode == "rejected" and reviews.get(sample["id"], {}).get("decision") != "rejected":
                        continue
                    if filter_mode == "compare" and sample["id"] not in predictions:
                        continue
                    payload.append(self._sample_payload(sample))
                self._send_json({
                    "items": payload,
                    "summary": {
                        "total": len(samples),
                        "reviewed": len(reviews),
                        "approved": sum(1 for r in reviews.values() if r["decision"] == "approved"),
                        "rejected": sum(1 for r in reviews.values() if r["decision"] == "rejected"),
                        "with_prediction": sum(1 for s in samples if s["id"] in predictions),
                    },
                })
                return

            if parsed.path == "/api/image":
                query = parse_qs(parsed.query)
                raw_path = query.get("path", [""])[0]
                image_path = Path(unquote(raw_path))
                self._send_file(image_path)
                return

            if parsed.path == "/" or parsed.path == "/index.html":
                self._send_file(static_dir / "index.html", "text/html; charset=utf-8")
                return

            if parsed.path == "/app.js":
                self._send_file(static_dir / "app.js", "text/javascript; charset=utf-8")
                return

            if parsed.path == "/styles.css":
                self._send_file(static_dir / "styles.css", "text/css; charset=utf-8")
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            sample_id = payload["id"]
            decision = payload["decision"]
            note = payload.get("note", "").strip()

            if decision not in {"approved", "rejected"}:
                self._send_json({"error": "invalid decision"}, status=400)
                return

            reviews[sample_id] = {
                "id": sample_id,
                "decision": decision,
                "note": note,
                "updated_at": datetime.now().isoformat(),
            }
            persist_reviews(review_path, reviews)
            self._send_json({"ok": True, "review": reviews[sample_id]})

        def log_message(self, fmt: str, *args) -> None:
            return

    return ReviewHandler


def serve(args) -> None:
    """启动本地审批服务。args 为命令行解析后的 Namespace。"""
    input_path = Path(args.input)
    review_path = Path(args.reviews)
    static_dir = PROJECT_ROOT / "review" / "ui"

    samples = load_samples(input_path)
    predictions = load_predictions(Path(args.predictions)) if getattr(args, "predictions", None) else {}
    handler = make_handler(samples, review_path, static_dir, predictions)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"review_items: {len(samples)}")
    print(f"predictions_loaded: {len(predictions)}（学生答案对比；0 表示未传 --predictions 或文件为空）")
    print(f"review_file: {review_path}")
    print(f"url: http://{args.host}:{args.port}")
    server.serve_forever()
