# 详细使用指南

遥感 VQA 蒸馏 → 人工审核 → Qwen3-VL LoRA 微调的全流程说明。
每个入口脚本在本文只出现一次，按运行顺序排列。

## 目录

1. [总览：流程与入口](#总览流程与入口)
2. [命令速查（从零到训练集）](#命令速查从零到训练集)
3. [起 vLLM 服务](#起-vllm-服务)
4. [① 构造数据 `01_generate.py`](#-构造数据-01_generatepy)
5. [② 数据体检 `src/health.py`](#-数据体检-srchealthpy)
6. [③ 切训练 / 评测集 `03_convert.py`](#-切训练--评测集-03_convertpy)
7. [④ 基线预测 `05_baseline.py`](#-基线预测-05_baselinepy)
8. [⑤ 人工审核 + 多模型对比 `02_review.py`](#-人工审核--多模型对比-02_reviewpy)
9. [⑥ SFT 训练 `04_train.py`](#-sft-训练-04_trainpy)
10. [训练后：推理 / 合并 `07` / `06`](#训练后推理--合并-07--06)
11. [常见问题](#常见问题)
12. [高级用法](#高级用法)

---

## 总览：流程与入口

```
起服务(teacher:8001 + baseline:8002)
   │
   ▼
① 01_generate.py   构造数据(描述→出题→蒸馏答案，三阶段并行) + 自动体检
   │  └─ outputs/03_answers_sft.jsonl
② src/health.py    数据体检(只读，红线告警)               ← ①末尾自动跑
   │
③ 03_convert.py    切 train/val + 路径重映射 + 转 Unsloth 可读
   │  └─ outputs/sft/{train,val}.jsonl
④ 05_baseline.py   未微调模型在 val 上的预测，作对照
   │  └─ outputs/baseline/val_pred.jsonl
⑤ 02_review.py     人工审核 + 多模型并排对比(浏览器)
   │  └─ outputs/manual_reviews.jsonl
⑥ 04_train.py      Qwen3-VL + LoRA 微调(仅 Linux+GPU)
      └─ outputs/qwen3vl-sft-lora  → 07_infer_lora.py / 06_merge_lora.py
```

**所有 vLLM 服务都在 `10.129.107.145`**（不是 localhost），脚本默认 `--base-url` 已指向它：

| 端口 | 模型 | 用途 |
|---|---|---|
| `8001` | Qwen3.5-27B | 教师，造数据 |
| `8002` | 未微调 Qwen3-VL-2B | 基线，做对照 |

> 图片路径也是这台 Linux 机上的 `/home/charles/mycode/sft+rl/dataset/...`，因此调用 vLLM、读图的脚本（①③④）要在这台机上跑。

---

## 命令速查（从零到训练集）

前提：图像数据集在 `--dataset-root`（默认 `dataset/`）下，在那台 Linux 机上执行。每步细节见对应章节。

```bash
# Step 0. 起服务：教师(8001) + 未微调基线(8002)，命令见「起 vLLM 服务」

# ① 构造数据(三阶段并行)，结束后自动出一次体检报告
python 01_generate.py --datasets "LEVIR-CD+,SECOND" --num-images 1000

# ② 体检(①末尾已自动跑；想单独再看就这条)
python -c "from src import health; health.report(health.load('outputs/03_answers_sft.jsonl'))"

# ③ 切 train/val（baseline 和训练都读它，所以排在前面）
python 03_convert.py --val-ratio 0.05 --image-root /your/abs/path/to/dataset

# ④ 未微调模型在 val 上的基线预测
python 05_baseline.py            # 读 val.jsonl、打 8002、写 outputs/baseline/val_pred.jsonl

# ⑤ 人工审核 + 多模型对比(浏览器 http://127.0.0.1:8008)
python 02_review.py \
  --predictions 2B未微调=outputs/baseline/val_pred.jsonl \
  --predictions 教师=outputs/baseline/val_pred2.jsonl \
  --predictions 2B微调后=outputs/baseline/val_pred3.jsonl

# ⑥ SFT 训练(仅 Linux+GPU；超参改脚本顶部常量)
python 04_train.py
```

**几个要点：**

- **`--num-images` 是"抽几张图"，不是"生成几条样本"。** 每张图出 2–5 问、每问 1 答，样本数 ≈ 图数 × 3 上下。
- **③ 排在 ④ 之前**：baseline(④) 和训练(⑥) 都读 `outputs/sft/{val,train}.jsonl`，所以先切。
- **体检是诊断不是闸门**：真正的质量闸是 ⑤ 的人工审核；红线只卡分布塌缩。
- **训完做"未微调 vs 微调"对比**：把合并后的学生权重另起一个服务，`python 05_baseline.py --base-url http://10.129.107.145:<port>/v1` 重出预测，再把多个预测文件一起喂给 ⑤。

---

## 起 vLLM 服务

### 教师模型（Qwen3.5-27B，端口 8001）

造数据用。Qwen3.5-27B / Qwen3-VL 系列，2 卡 tensor parallel（实测可用）：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code \
  --dtype bfloat16 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --tensor-parallel-size 2 \
  --port 8001
```

| 参数 | 作用 |
|---|---|
| `--mm-encoder-tp-mode data` | 视觉编码器用 data parallel，多卡吞吐更高 |
| `--mm-processor-cache-type shm` | 图像预处理走共享内存缓存，重复访问同图命中 |
| `--reasoning-parser qwen3` | 启用 Qwen3 思考链解析（流水线侧已 `enable_thinking=False` 关思考，避免输出截断） |
| `--enable-prefix-caching` | 相同前缀 prompt 复用 KV cache，对本项目高度重复的模板显著提速 |
| `--max-model-len 8192` | **总长上限(图像 token+prompt+输出)，勿低于 8192**。遥感图编码常占 1000~3000+ token，设 4096 会挤压输出导致答案截断（见 [数据问题记录 #7](DATA_QUALITY_LOG.md)）。脚本侧 `max_tokens=2048` 与之配合 |
| `--gpu-memory-utilization 0.9` | 留 10% 显存给峰值，避免 OOM |

> 流水线默认 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 关闭思考，正常不会再出现 `finish_reason=length` 把 token 全花在 reasoning 上。
>
> 备选：`swift deploy --model Qwen/Qwen3.5-27B --port 8001 --infer-backend vllm`。

### 基线小模型（未微调 Qwen3-VL-2B，端口 8002）

给微调模型做对照：未微调原始 Qwen3-VL-2B（与学生同款基座）单独起在 8002，供 `05_baseline.py` 跑基线预测。2B 很小，单卡足够、显存占比给低（可与别的服务共卡）：

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve /home/charles/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct \
  --served-model-name Qwen3-VL-2B-Instruct \
  --trust-remote-code \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.7 \
  --max-model-len 8192 \
  --enforce-eager \
  --port 8002
```

> - **`--enforce-eager` 必加**：关 CUDA graph，避免 Qwen3-VL 在 V1 引擎下混批触发 deepstack 缓冲越界（`ValueError: Requested more deepstack tokens than available in buffer`）导致 EngineCore fatal、整个服务挂。小批量评测已验证可跑；慢一点但最省心。大批量可改用 `--no-enable-chunked-prefill`（保留 graph、只禁混批）。
> - 模型名不用记：`05_baseline.py` / `VLMClient` 会自动探测 `/v1/models` 返回的名字。
> - 训练完成后，把**合并后的学生权重**用同样命令换路径起一个服务（端口随意），即可再跑一遍 `05_baseline.py` 得到"微调学生"的预测。

---

## ① 构造数据 `01_generate.py`

单一入口，内部三阶段并行推进（逻辑在 `src/generation.py`）：描述 → 出题 → 蒸馏答案，结束后自动跑一次只读体检。三个阶段不能单独运行，参数全挂在 `01_generate.py` 上。

```bash
# 默认：LEVIR-CD+ 500 + SECOND 500，每图 2–5 问
python 01_generate.py

# 小规模冒烟：每数据集各抽 2 张
python 01_generate.py --samples-per-dataset 2

# 全局混合随机采样 + 纳入 EBD
python 01_generate.py --datasets "EBD,LEVIR-CD+,SECOND" --num-images 50

# 跳过结束后的自动体检
python 01_generate.py --no-health-check
```

### 参数

完整见 `python 01_generate.py -h`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | 自动探测 | 模型名，如 `Qwen/Qwen3-VL-32B-Instruct`；见 [模型名探测](#模型名探测) |
| `--base-url` | `http://10.129.107.145:8001/v1` | 教师 vLLM 服务地址 |
| `--retries` | `3` | 单条 API 失败最大重试次数 |
| `--dataset-root` | `dataset` | 数据集根目录 |
| `--datasets` | `LEVIR-CD+,SECOND` | 要处理的数据集，逗号分隔 |
| `--num-images` | `None` | 全局混合随机采样总数；传入后优先于 `--samples-per-dataset` |
| `--samples-per-dataset` | `LEVIR-CD+=500,SECOND=500` | 每数据集采样数；也可写单一整数对所有数据集生效 |
| `--ebd-per-disaster` | `None` | EBD 按 7 种灾害子目录各抽 N 张 |
| `--seed` | `42` | 采样 + 软配额分配的随机种子 |
| `--manifest` / `--refresh-manifest` | — | 复用 / 强制重抽样本清单 |
| `--num-questions` / `--min-questions` | `5` / `2` | 每张图问题数上 / 下限（软配额在区间内分配类型） |
| `--max-concurrency` | `24` | 三阶段共用的最大并发请求数 |
| `--output-dir` | `outputs` | 输出目录（描述/问题/答案/stats/manifest 都落这里） |
| `--no-health-check` | 关 | 跳过生成结束后的自动体检 |
| `--direct-ratio` | `0.3` | **已废弃**：纯蒸馏不分 analysis/direct，不再生效（仅兼容旧命令行） |

> 各阶段 temperature 在 `src/prompts.py` / 对应脚本里：描述 0.3、出题 0.4、答案 0.3。

### 默认采样规模

| 数据集 | 数量 | 说明 |
|---|---|---|
| LEVIR-CD+ | 500 | 城市建筑与道路 |
| SECOND | 500 | 多类地表覆盖 |
| **合计** | **1000 张** | 每图 2–5 问，预计 **2000–5000 个 VQA 样本** |

默认不启用 EBD（灾后场景）。需要时显式传 `--datasets "LEVIR-CD+,SECOND,EBD" --samples-per-dataset "LEVIR-CD+=500,SECOND=500,EBD=140" --ebd-per-disaster 20`。

### 运行行为

- **问题数自适应**：`2–5` 区间内由模型按图像内容决定——简单图（纯水体/单一植被）2 个，丰富图最多 5 个，超出硬截断到 5，不会凑数重复。
- **三阶段调度**：Phase 1 出某图描述后立刻交 Phase 2，两者并行；Phase 3 等前两者全部完成后统一开始；三阶段共用一个并发上限（默认 24）。
- **断点续跑**：首次把采样固化到 `outputs/sample_manifest.jsonl`，后续默认复用做续跑，不重新随机；要重抽传 `--refresh-manifest`。续跑以**图像绝对路径**为唯一标识，移动数据集目录会让旧记录失效。
- 三阶段完成后自动跑一次只读体检（`--no-health-check` 可关）。

### 三阶段产出

**Phase 1 — 受控图像描述**（`outputs/01_descriptions.jsonl`）：教师先对每图生成保守客观的描述，作为后续问答的"证据边界"。

```json
{"image_path": "/path/to/image.png", "dataset": "LEVIR-CD+",
 "description": "1. 场景类型：城市建成区...", "generated_at": "..."}
```

**Phase 2 — 多样化问题**（`outputs/02_questions.jsonl`）：基于描述生成 7 类 VQA 问题（existence / location / spatial_relation / attribute / counting / unanswerable / ambiguous）。每图按**软配额**（`src/sampling.py`）注入"建议类型清单"，使全局类型比例贴近目标、不易塌缩；模型可对不适合的图跳过某类型。

```json
{"image_path": "...", "dataset": "LEVIR-CD+", "description": "...",
 "questions": [{"question_type": "existence", "question": "图像中是否有水体？"}]}
```

**Phase 3 — 纯蒸馏答案**（`outputs/03_answers_sft.jsonl`，最终 SFT 数据）：把"图像 + 问题"直接交给教师，**原样收下回答**——不套 `<analysis>`/`<answer>`、不分 analysis/direct、不做格式校验，蒸馏 prompt 就是原问题文本，连 Phase 1 描述也不传。

```json
{"messages": [
   {"role": "user", "content": "<image>\n图像中是否有水体？"},
   {"role": "assistant", "content": "有。图像左下角可见一片深色水域，边界清晰。"}],
 "images": ["/path/to/image.png"],
 "meta": {"task_type": "vqa_distill", "question_type": "existence", "source_dataset": "LEVIR-CD+"}}
```

> `assistant.content` 是教师原始自然回答，长度/风格由教师决定；`meta` 不含 `answer_mode` / `auto_validation`。

---

## ② 数据体检 `src/health.py`

发布前的数据治理：体检（只读，给报告 + 红线告警）+ 按配比降采样。体检由 `01_generate.py` 末尾**自动调用**；降采样在 `03_convert.py --rebalance` 时触发。

```bash
# 单独再看一次（配比/归一化熵/问题前缀塌缩/答案模板化/长度统计/编码自检）
python -c "from src import health; health.report(health.load('outputs/03_answers_sft.jsonl'))"
```

**红线判定**（`report()` 返回 True 即触发，`01_generate.py` 据此打印告警）：

| 维度 | 红线 |
|---|---|
| question_type 归一化熵 | < 0.7（类型分布失衡） |
| 问题前缀 Top-3 合计 | > 60%（Question Distribution Collapse） |
| 单一答案开头占比 | > 20%（过度模板化） |
| 答案长度 p90 | > 120（整体报告化） |

> `src/health.py` 已按纯蒸馏形态改写：问题侧关注分布与塌缩，答案侧只关注模板化和长度，不再评估拒答率或"是否按护栏回答"。

---

## ③ 切训练 / 评测集 `03_convert.py`

把给人看的 `03_answers_sft.jsonl` 转成 `04_train.py` 能直接读的 `train.jsonl` / `val.jsonl`（逻辑在 `src/convert.py`，Win/Linux 都能跑）。

```bash
python 03_convert.py \
  --image-root /your/abs/path/to/dataset \  # 训练机上 dataset 真实绝对路径，到 Linux 上必填
  --val-ratio 0.05 \
  --rebalance \                             # 可选：按配比降采样，主要把 counting 压到 5%
  --check-images                            # 图片在本机时建议开，校验每张图可打开
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--src` | `outputs/03_answers_sft.jsonl` | 输入答案 jsonl |
| `--out-dir` | `outputs/sft` | 输出目录 |
| `--old-root` | `/home/charles/mycode/sft+rl/dataset` | 被替换的原图片根目录 |
| `--image-root` | 空 | **训练机上的图片根目录**；留空不改路径（到 Linux 务必传） |
| `--val-ratio` | `0.02` | 验证集比例 |
| `--rebalance` | 关 | 转换前按目标配比降采样（主要压 counting）；建议人工审核之后再开 |
| `--check-images` | 关 | 校验每张图是否存在（需图片在本机） |
| `--keep-warning` | 关 | **纯蒸馏下基本 no-op**：`auto_validation` 已移除，过滤默认全保留 |

> 路径重映射**始终输出 POSIX 斜杠**，即便在 Windows 上跑，产出路径到 Linux 也直接可用。

### 为什么需要这一步？

生成格式（`messages` + 平行 `images` + `<image>` 占位符）是 **LLaMA-Factory / ShareGPT 风格**，LLaMA-Factory 能直接读，但 **Unsloth 用的是另一套（结构化 content）**，所以要转一次。差异：

| 维度 | 生成格式 | Unsloth 要的 | 不转的后果 |
|---|---|---|---|
| 图像位置 | `content` 里 `<image>` 占位符 + 独立 `images` 字段 | `content` 列表里的结构化图像元素 | `<image>` 被当普通文本，图喂不进 |
| `content` 类型 | 字符串 | 列表（text/image 元素） | collator 解析不到图像 token |
| 图片引用 | 文件路径字符串 | PIL/可解析图对象 | 需 `Image.open` 加载（脚本内做） |
| 路径根目录 | 生成机 `/home/charles/...` | 训练机真实路径 | 不重映射找不到图 |

Unsloth 在内存里构造的目标格式（`04_train.py` 自动做，不用手写）：

```python
[{"role": "user", "content": [
    {"type": "image", "text": None, "image": <PIL.Image>},
    {"type": "text",  "text": "图像中是否有水体？"}]},
 {"role": "assistant", "content": [{"type": "text", "text": "有。..."}]}]
```

---

## ④ 基线预测 `05_baseline.py`

用未微调原始模型在 val 上跑一遍，作为微调对照。发给基线的是**裸问题**（只有 question + 图，不是蒸馏 prompt），与微调学生的推理输入完全一致，对照才公平。支持并发 + 断点续跑（按 `image|question` 去重）。

```bash
python 05_baseline.py            # 读 val.jsonl、打基线 8002、写 outputs/baseline/val_pred.jsonl
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--src` | `outputs/sft/val.jsonl` | 测试集 jsonl（每行 `{image, question, answer, ...}`） |
| `--out` | `outputs/baseline/val_pred.jsonl` | 预测输出 |
| `--base-url` | `http://10.129.107.145:8002/v1` | 基线服务地址（换成微调学生服务即可生成学生预测） |
| `--model` | 自动探测 | 模型名 |
| `--max-concurrency` | `16` | 并发请求数 |
| `--max-tokens` | `1024` | 单条回答上限 |
| `--temperature` | `0.0` | 默认贪心，结果可复现 |
| `--limit` | `None` | 只跑前 N 条（冒烟用） |

输出每行：`{image, question, question_type, reference(教师答案), prediction(本模型回答)}`，直接喂给 ⑤ 的 `--predictions` 做对比。

---

## ⑤ 人工审核 + 多模型对比 `02_review.py`

SFT 数据进训练前的最后一步、也是必走一步——自动质检只能拦格式与硬规则错误，幻觉/错答/计数误差只能靠人工。轻量本地 Web 工具（逻辑在 `src/review.py`），两个模式：① 人工审批蒸馏数据，② 多模型输出对比。

```bash
# 最简：全用默认值，仅人工审批
python 02_review.py

# 带多模型对比：--predictions 可多次传入，每次 "标签=路径"，前端按顺序每模型一列并排
python 02_review.py \
  --predictions 2B未微调=outputs/baseline/val_pred.jsonl \
  --predictions 教师=outputs/baseline/val_pred2.jsonl \
  --predictions 2B微调后=outputs/baseline/val_pred3.jsonl
# 浏览器打开 http://127.0.0.1:8008，Ctrl+C 停止
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | `outputs/03_answers_sft.jsonl` | 要审核的 SFT JSONL |
| `--reviews` | `outputs/manual_reviews.jsonl` | 审核结果落盘（不存在自动创建） |
| `--predictions` | 无 | 模型预测 JSONL（`05_baseline.py` 风格），写成 `标签=路径`，**可多次传入**；只给路径时用文件名当标签 |
| `--host` | `127.0.0.1` | 监听地址；要别的机器访问改 `0.0.0.0`（无鉴权，注意网络隔离） |
| `--port` | `8008` | 监听端口 |

### 界面与行为

- **模式① 人工审批**：逐条 `通过 / 不通过` + 备注，可按 `未审批 / 全部 / 仅通过 / 仅不通过` 筛选。结果**幂等覆盖**写入 `manual_reviews.jsonl`，同一样本以最后一次为准。
- **模式② 多模型对比**：按 `--predictions` 顺序每模型一列并排（按 `image|question` 对齐）。自动只显示"至少有一个模型出了预测"的样本；某模型对某条无预测则该列显示「（无预测）」。不传 `--predictions` 时模式②为空，模式①不受影响。
- **样本主键** = `image_path | question`：图片路径或问题文本一变，旧审批记录就"对不上"被视为未审。
- **新增样本要重启**：启动时一次性加载，运行期不重读 `--input`。
- **图片按 `images[0]` 路径直接读**，相对路径相对于启动目录——建议在项目根启动。

### 下游消费

把 `manual_reviews.jsonl` 与 `03_answers_sft.jsonl` 按 `id`（即 `image_path|question`）join，仅留 `decision == "approved"`，即得人工审核通过的集合用于训练。

---

## ⑥ SFT 训练 `04_train.py`

用 [Unsloth](https://unsloth.ai/docs) 对 Qwen3-VL 做视觉 LoRA 微调，**仅 Linux+GPU**。`04_train.py` 是**扁平脚本**（照搬官方 Qwen3-VL Vision notebook 结构，刻意不封装、无命令行参数），**改超参直接改脚本顶部各 section 常量**。

### 装依赖

```bash
pip install -r requirements-train.txt   # 先按 Unsloth 官方指引装好匹配的 torch+CUDA
swanlab login                           # 训练用 SwanLab 看 loss/lr/显存曲线，首次登录一次
```

> `04_train.py` 已接入 `SwanLabCallback`，自动推曲线到看板（项目/实验名见脚本顶部 `SWANLAB_PROJECT` / `SWANLAB_EXPERIMENT`）。想纯离线给回调加 `mode="local"`。

### 运行

```bash
# 冒烟：取消脚本里 `# max_steps = 30` 那行注释(只跑 30 步)，确认数据/显存/模板都通
python 04_train.py

# 正式：把 max_steps 注释回去，按需调 num_train_epochs 等常量，再跑
python 04_train.py
```

脚本分六步，参数就近注释：

| 步骤 | 关键常量 | 说明 |
|---|---|---|
| ① 加载模型 | `MODEL_NAME` / `load_in_4bit` | 基座 Qwen3-VL-2B（默认 16bit LoRA）；显存紧设 `load_in_4bit=True` 走 QLoRA |
| ② 挂 LoRA | `r` / `lora_alpha` / `finetune_vision_layers` | 见下「起步配方」 |
| ③ 准备数据 | `convert_to_conversation` | 每行 `{image,question,answer}` 转 Unsloth 对话 + `Image.open` 读图 |
| ④ 训练器 | `SFTConfig(...)` | 批大小/时长/学习率/验证/取最佳 |
| ⑤ 训练 | — | `trainer.train()` |
| ⑥ 保存 | `OUTPUT_DIR` | 存 LoRA；末尾有合并 16bit 的可选注释行 |

- 用 `FastVisionModel` + `UnslothVisionDataCollator`，**只训 LoRA 适配器**，基座冻结。
- 数据是 eager 列表（一次读进内存），几千张没问题；规模很大时改 `with_transform` 惰性读图（脚本有注释）。

### 起步配方（~3700 条规模，抗遗忘版）

约 3700 条训练样本，**有效 batch = `per_device_train_batch_size` × `gradient_accumulation_steps` = 16**，每 epoch ≈ 230 步。窄数据（单一遥感 VQA）上，太大的 lr/容量/训练量会把 2B 整体往这套数据上拽、牺牲通用能力，故默认按抗遗忘调过：

| 常量 | 默认 | 抗遗忘考量 |
|---|---|---|
| `num_train_epochs` | `1` | 窄数据 1 epoch 往往够，配合"取最佳"防过拟合 |
| `per_device_train_batch_size` / `gradient_accumulation_steps` | `2` / `8` | 有效 batch 16；OOM 就改 1 / 16，乘积不变 |
| `learning_rate` | `5e-5` | **从 2e-4 降下来**：2e-4 在窄数据 1 epoch 就推得很远，是遗忘主因 |
| `r` | `8` | **从 16 降到 8**：直接砍可改写子空间，抗遗忘最直接旋钮；欠拟合再升回 16/32 |
| `lora_alpha` | `16` | 等效放大 alpha/r = 2.0；想更抗遗忘降到 8（ratio=1），代价学得慢 |
| `lora_dropout` | `0.05` | 一点正则 |
| `finetune_vision_layers` | `False` | **关视觉塔**：3700 张遥感图训它易破坏对自然图理解，本任务也不靠学新视觉特征 |
| `load_in_4bit` | `False` | 16bit 普通 LoRA；显存紧设 True 走 QLoRA |

验证与"取最佳"（已默认开启，抗遗忘关键，均在 `SFTConfig` 内）：

| 常量 | 默认 | 作用 |
|---|---|---|
| `eval_strategy` / `eval_steps` | `"steps"` / `10` | 每 10 步在 val 上算 loss，画出过拟合拐点 |
| `load_best_model_at_end` | `True` | 训练结束**回滚到 val loss 最低的 checkpoint** |
| `metric_for_best_model` / `greater_is_better` | `"eval_loss"` / `False` | 以验证 loss 选最佳 |
| `save_strategy` / `save_steps` | `"steps"` / `10` | 必须与 eval 对齐，否则最佳点可能没存下 |

> `save_steps=10` 存得勤，靠 `save_total_limit=3` 控盘（LoRA 才几十 MB）。当前**未开早停**——`load_best_model_at_end` 只回滚不提前停；要早停加 `EarlyStoppingCallback`。仍嫌遗忘重，最治本的是**混入回放数据**（5%~20% 通用图文/纯文本指令），比调超参更有效。

---

## 训练后：推理 / 合并 `07` / `06`

| 目的 | 用 | 命令 |
|---|---|---|
| 本地快速验效果 | `07_infer_lora.py`（直接用 LoRA adapter） | 见下 |
| 部署 / 交付完整模型 | `06_merge_lora.py`（合并成 16bit） | 见下 |
| 比较不同 checkpoint | 两者都把 `--adapter` 指向某个 `checkpoint-*` 即可 | — |

```bash
# 方案一：直接用 adapter 推理
python 07_infer_lora.py \
  --adapter outputs/qwen3vl-sft-lora \
  --image /home/charles/mycode/sft+rl/dataset/LEVIR-CD+/train/B/train_369.png \
  --question "图像右上角建筑物的屋顶呈现什么颜色？"

# 方案二：合并成完整 16bit 模型目录
python 06_merge_lora.py \
  --adapter outputs/qwen3vl-sft-lora \
  --out outputs/qwen3vl-sft-lora_merged
```

`07_infer_lora.py` 常用参数：`--max-new-tokens 256`、`--temperature 0.0`（>0 才采样）、`--min-p 0.1`、`--load-in-4bit`（仅 4bit 流程开）。
`06_merge_lora.py`：`--out` 默认 `<adapter>_merged`，`--load-in-4bit` 仅 4bit 流程开。

---

## 常见问题

### 模型名探测

调用 vLLM 的脚本启动时自动调 `/v1/models` 探测服务端实际加载的模型名，**最推荐**，避免名称不匹配。优先级：

```
用户 --model > 环境变量 VLLM_MODEL > 自动探测 > 默认值
```

```bash
curl -s http://10.129.107.145:8001/v1/models | python -m json.tool   # 看实际加载的模型
```

### 连接 vLLM 失败

```bash
curl http://10.129.107.145:8001/v1/models          # 确认服务在
python 01_generate.py --base-url http://localhost:8000/v1   # 确认端口
```

### 断点续传不生效

依赖**图像绝对路径**作唯一标识。移动数据集目录会让旧记录失效——删输出文件重跑，或手动过滤已处理的。

### 教师答案质量不稳定 / 偶尔跑题

纯蒸馏不做格式校验，质量取决于教师（Qwen3.5-27B），`meta` 也无 `auto_validation`。把关靠 [人工审核](#-人工审核--多模型对比-02_reviewpy)——发现系统性问题就回去改问题分布、换更强教师，或重定义蒸馏策略。

---

## 高级用法

### 分数据集并行处理

用不同 `--output-dir` 把各数据集隔离，互不干扰、可同时跑：

```bash
python 01_generate.py --datasets EBD       --num-images 50 --output-dir outputs/ebd
python 01_generate.py --datasets LEVIR-CD+ --num-images 50 --output-dir outputs/levir
```

### 自定义 Prompt 模板

编辑 `src/prompts.py` 中的模板（如 `DESCRIPTION_PROMPT`）。注：纯蒸馏的 `ANSWER_DISTILL_PROMPT` 就是原问题文本本身，不主动约束教师——若要"纯模仿教师分布"的 SFT，不建议往这里加包装语或护栏；想换风格直接换 `--model` / `--base-url`。`src/validator.py` 那套旧格式校验已不再调用。

### 添加新数据集

编辑 `src/scanner.py` 的 `DATASET_CONFIGS`：

```python
DATASET_CONFIGS = {
    "MY_DATASET": {
        "glob_patterns": ["**/images/*.png"],
        "image_filter": lambda p: True,
    },
}
```

### 快速抽看数据质量

```bash
python -c "
import json
with open('outputs/03_answers_sft.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        r = json.loads(line)
        print('Q:', r['messages'][0]['content'].replace('<image>','').strip())
        print('A:', r['messages'][1]['content'])
        print('-' * 40)
"
```
