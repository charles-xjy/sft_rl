#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学习用脚本 01:查看 Qwen3-VL-4B 的 chat template,搞懂"数据到底变成什么样"。

训练前,你的一条样本会经过这条流水线:
    原始 jsonl
      -> convert()       把 <image> token / 路径,转成结构化 messages
      -> chat template   把 messages 渲染成一段带特殊 token 的纯文本   <-- 本脚本看这步
      -> tokenizer       纯文本切成 token id
      -> 训练            只在 assistant 段算 loss

chat template 是上面第 3 步的"规则",决定了 <|im_start|>、<|image_pad|> 这些
特殊标记怎么插。看懂它,就看懂了模型训练时实际"读到"的是什么。

在能联网/有模型的服务器上跑:
    python scratch/learn/01_inspect_template.py
    python scratch/learn/01_inspect_template.py --data outputs/03_answers_sft.jsonl --n 2

依赖: transformers
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--data", default=None,
                   help="给出 jsonl 路径则用真实样本渲染;不给则用内置示例")
    p.add_argument("--n", type=int, default=1, help="渲染前 N 条真实样本")
    args = p.parse_args()

    from transformers import AutoProcessor

    print(f"\n{'='*70}\n加载处理器: {args.model}\n{'='*70}")
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer

    # ----- 1. 模板源码(Jinja2) ----- #
    template = getattr(proc, "chat_template", None) or getattr(tok, "chat_template", None)
    print(f"\n{'='*70}\n[1] chat template 源码 (Jinja2)\n{'='*70}")
    if template:
        print(template)
    else:
        print("（处理器没挂 chat_template，可能在 tokenizer_config.json 里，或需手动指定）")

    # ----- 2. assistant_only_loss 能否用 ----- #
    print(f"\n{'='*70}\n[2] 关键检查\n{'='*70}")
    has_gen = template and "generation" in template
    print(f"模板含 {{% generation %}} 标记 : {has_gen}"
          f"   -> assistant_only_loss {'可直接用' if has_gen else '需 TRL 打补丁/确认'}")
    print(f"eos_token : {tok.eos_token!r}")
    print(f"pad_token : {tok.pad_token!r}")
    for name in ("image_token", "vision_start_token", "image_token_id"):
        if hasattr(proc, name) or hasattr(tok, name):
            val = getattr(proc, name, None) or getattr(tok, name, None)
            print(f"{name} : {val!r}")

    # ----- 3. 用样本渲染,看最终纯文本 ----- #
    samples = load_samples(args.data, args.n)
    print(f"\n{'='*70}\n[3] 渲染结果（模型训练时实际读到的纯文本）\n{'='*70}")
    for i, msgs in enumerate(samples):
        print(f"\n----- 样本 {i} -----")
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        print(text)

        # 编码自检:打印是否含可疑乱码(常见 CJK 乱码区/替换符)
        suspicious = sum(1 for ch in text if ch in "锘" or 0xE000 <= ord(ch) <= 0xF8FF)
        print(f"[编码自检] 长度 {len(text)} 字符；可疑乱码字符 {suspicious} 个"
              + ("  <-- 注意！文本可能是乱码" if suspicious else "  (看起来正常)"))

        # token 数量(VLM 里图像会展开成很多 image_pad token,这里只是文本侧粗看)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        print(f"[token] 文本 token 数 ≈ {len(ids)}（不含图像 patch 展开）")


def load_samples(data_path, n):
    """从 jsonl 读前 n 条并转成结构化 messages;无 data 则给内置示例。"""
    if not data_path:
        return [[
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "图像右侧道路附近是否分布有建筑？"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text":
                 "<analysis>\n1. 定位：图像右侧道路附近。\n2. 观察：可见规则矩形屋顶。\n"
                 "3. 判断：符合建筑分布特征。\n</analysis>\n<answer>是，分布有多个建筑。</answer>"},
            ]},
        ]]

    out = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = []
            for m in rec["messages"]:
                raw = m["content"]
                n_img = raw.count("<image>")
                text = raw.replace("<image>\n", "").replace("<image>", "").strip()
                content = [{"type": "image"} for _ in range(n_img)]
                if text:
                    content.append({"type": "text", "text": text})
                msgs.append({"role": m["role"], "content": content})
            out.append(msgs)
            if len(out) >= n:
                break
    return out


if __name__ == "__main__":
    main()
