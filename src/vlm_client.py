"""
VLM客户端封装
"""
import base64
import os
import re
from pathlib import Path

import requests
from openai import OpenAI


def encode_image(image_path: Path) -> str:
    """将图像编码为base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_model(base_url: str) -> str:
    """
    自动探测vLLM服务端加载的模型名称

    Args:
        base_url: vLLM服务地址，如 http://10.129.107.145:8001/v1

    Returns:
        探测到的模型名称，探测失败返回None
    """
    try:
        # 处理base_url可能带或不带/v1后缀的情况
        models_url = re.sub(r'/v1/?$', '', base_url) + '/v1/models'
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            return data['data'][0]['id']
    except Exception as e:
        print(f"[警告] 自动探测模型失败: {e}")
    return None


class VLMClient:
    """VLM客户端封装，支持重试机制和自动模型探测"""

    def __init__(self, base_url: str = None, model: str = None, retries: int = 3):
        self.base_url = base_url or "http://10.129.107.145:8001/v1"
        self.retries = retries

        # 模型名称优先级: 用户指定 > 环境变量 > 自动探测 > 默认值
        if model:
            self.model = model
        elif os.environ.get("VLLM_MODEL"):
            self.model = os.environ.get("VLLM_MODEL")
        else:
            detected = detect_model(self.base_url)
            if detected:
                self.model = detected
                print(f"[自动探测] 服务端模型: {detected}")
            else:
                self.model = "Qwen/Qwen3-VL-32B-Instruct"
                print(f"[警告] 使用默认模型: {self.model}")

        self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)

    def call(self, prompt: str, image_path: Path, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """调用VLM模型，带重试机制"""
        base64_img = encode_image(image_path)

        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                        ],
                    }],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.retries - 1:
                    raise
                continue
        return ""
