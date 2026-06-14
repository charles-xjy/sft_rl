#!/usr/bin/env python3
"""
Local review server for manual approve/reject of generated VQA samples.
"""
import argparse
import json
import mimetypes
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


def load_samples(input_path: Path) -> list[dict]:
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


def load_reviews(review_path: Path) -> dict[str, dict]:
    reviews: dict[str, dict] = {}
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


def persist_reviews(review_path: Path, reviews: dict[str, dict]) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as f:
        for review in reviews.values():
            f.write(json.dumps(review, ensure_ascii=False) + "\n")


def make_handler(samples: list[dict], review_path: Path, static_dir: Path):
    reviews = load_reviews(review_path)

    class ReviewHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict | list, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path, content_type: str | None = None) -> None:
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
                    payload.append(self._sample_payload(sample))
                self._send_json({
                    "items": payload,
                    "summary": {
                        "total": len(samples),
                        "reviewed": len(reviews),
                        "approved": sum(1 for r in reviews.values() if r["decision"] == "approved"),
                        "rejected": sum(1 for r in reviews.values() if r["decision"] == "rejected"),
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


def main():
    parser = argparse.ArgumentParser(description="Start local review UI")
    parser.add_argument("--input", default="outputs/03_answers_sft.jsonl", help="SFT JSONL file to review")
    parser.add_argument("--reviews", default="outputs/manual_reviews.jsonl", help="Review output JSONL path")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8008, help="Port to bind")
    args = parser.parse_args()

    input_path = Path(args.input)
    review_path = Path(args.reviews)
    static_dir = Path(__file__).parent / "review" / "ui"

    samples = load_samples(input_path)
    handler = make_handler(samples, review_path, static_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"review_items: {len(samples)}")
    print(f"review_file: {review_path}")
    print(f"url: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
