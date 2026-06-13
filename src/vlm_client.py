"""
VLM client wrapper.
"""
import base64
import json
import mimetypes
import os
import re
from pathlib import Path

import requests
from openai import OpenAI


def encode_image(image_path: Path) -> str:
    """Encode an image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_mime_type(image_path: Path) -> str:
    """Return a best-effort mime type for the input image."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/jpeg"


def extract_text_content(content) -> str:
    """Normalize OpenAI-compatible content blocks into plain text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                text = block
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
            else:
                text = getattr(block, "text", None) or getattr(block, "content", None) or ""

            if isinstance(text, list):
                text = extract_text_content(text)
            if text:
                parts.append(str(text).strip())
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def format_response_debug(response) -> str:
    """Serialize a response object for debug printing."""
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(), ensure_ascii=False, indent=2)
    if hasattr(response, "to_dict"):
        return json.dumps(response.to_dict(), ensure_ascii=False, indent=2)
    return str(response)


def detect_model(base_url: str) -> str:
    """Automatically detect the served model name from a vLLM endpoint."""
    try:
        models_url = re.sub(r"/v1/?$", "", base_url) + "/v1/models"
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["id"]
    except Exception as e:
        print(f"[warning] failed to detect model automatically: {e}")
    return None


class VLMClient:
    """VLM client wrapper with retries and response validation."""

    def __init__(self, base_url: str = None, model: str = None, retries: int = 3):
        self.base_url = base_url or "http://10.129.107.145:8001/v1"
        self.retries = retries

        if model:
            self.model = model
        elif os.environ.get("VLLM_MODEL"):
            self.model = os.environ.get("VLLM_MODEL")
        else:
            detected = detect_model(self.base_url)
            if detected:
                self.model = detected
                print(f"[auto-detect] service model: {detected}")
            else:
                self.model = "Qwen/Qwen3-VL-32B-Instruct"
                print(f"[warning] fallback model name: {self.model}")

        self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)

    def call_raw(self, prompt: str, image_path: Path, max_tokens: int = 1024, temperature: float = 0.3):
        """Call the VLM and return the raw OpenAI-compatible response."""
        base64_img = encode_image(image_path)
        mime_type = detect_mime_type(image_path)

        return self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}},
                ],
            }],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def call(
        self,
        prompt: str,
        image_path: Path,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        dump_response: bool = False,
    ) -> str:
        """Call the VLM and return normalized text output."""
        for attempt in range(self.retries):
            try:
                response = self.call_raw(
                    prompt=prompt,
                    image_path=image_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                if dump_response:
                    print("\n[raw-model-response]")
                    print(format_response_debug(response))
                    print("[/raw-model-response]\n")

                if not getattr(response, "choices", None):
                    raise ValueError("model returned no choices")

                choice = response.choices[0]
                message = getattr(choice, "message", None)
                text = extract_text_content(getattr(message, "content", None))
                if text:
                    return text

                finish_reason = getattr(choice, "finish_reason", "unknown")
                reasoning = extract_text_content(getattr(message, "reasoning_content", None))
                extra = f", reasoning_preview={reasoning[:160]!r}" if reasoning else ""
                raise ValueError(f"empty model response: finish_reason={finish_reason}{extra}")
            except Exception:
                if attempt == self.retries - 1:
                    raise
                continue
        return ""
