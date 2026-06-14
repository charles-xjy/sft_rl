# 详细使用指南

## 目录
1. [模型配置](#模型配置)
2. [生成流程（01_generate.py）](#生成流程01_generatepy)
3. [参数详解](#参数详解)
4. [常见问题排查](#常见问题排查)
5. [高级用法](#高级用法)
6. [完整运行流程（从零到训练集）](#完整运行流程从零到训练集)
7. [数据体检与降采样](#数据体检与降采样)
8. [人工审批前端](#人工审批前端)
9. [SFT 训练（Unsloth / Qwen3-VL）](#sft-训练unsloth--qwen3-vl)

---

## 模型配置

### 方式零：自动探测（默认行为 ✨）

脚本启动时会自动调用 vLLM 的 `/v1/models` 接口，探测服务端实际加载的模型名称：

```bash
# 不需要指定 --model，自动探测
python 01_generate.py
```

运行时会显示：
```
[自动探测] 服务端模型: Qwen/Qwen3-VL-32B-Instruct
```

这是**最推荐**的方式，避免模型名称不匹配问题。

---

### 方式一：通过命令行参数指定

生成入口 `01_generate.py` 支持 `--model` 和 `--base-url` 参数：

```bash
python 01_generate.py \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --base-url http://10.129.107.145:8001/v1
```

### 方式二：通过环境变量设置

```bash
# 设置环境变量
export VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct

# 脚本会自动读取
python 01_generate.py
```

### 模型名称优先级

```
用户指定 --model > 环境变量 VLLM_MODEL > 自动探测 > 默认值
```

---

## 启动vLLM服务

### 推荐配置（实测可用）

适用于 Qwen3.5-27B / Qwen3-VL 系列，2 卡 tensor parallel：

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

关键参数：

| 参数 | 作用 |
|---|---|
| `--mm-encoder-tp-mode data` | 多模态视觉编码器使用 data parallel，多卡下吞吐更高 |
| `--mm-processor-cache-type shm` | 图像预处理结果走共享内存缓存，重复访问同图时命中 |
| `--reasoning-parser qwen3` | 启用 Qwen3 思考链解析；流水线侧可通过 `enable_thinking=False` 关掉思考避免输出截断 |
| `--enable-prefix-caching` | 相同前缀的 prompt 复用 KV cache，对模板高度重复的本项目显著提速 |
| `--max-model-len 8192` | **总长度上限(图像 token + prompt + 输出),勿低于 8192**。遥感图编码常占 1000~3000+ token，若设成 4096 会挤压输出空间，导致答案被截断/漏 `</answer>`（见 [数据问题记录 #7](DATA_QUALITY_LOG.md)）。脚本侧 `max_tokens` 已设 2048 与之配合 |
| `--gpu-memory-utilization 0.9` | 留 10% 显存给峰值，避免 OOM |

> 流水线已默认通过 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 关闭思考模式，正常情况下不会再出现 `finish_reason=length` 把 4096 tokens 全花在 reasoning 上的问题。详见 [常见问题排查](#常见问题排查)。

### 使用Swift部署（备选）

```bash
swift deploy --model Qwen/Qwen3.5-27B \
  --port 8001 \
  --infer-backend vllm
```

### 多卡部署示例

```bash
# 4卡部署
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code \
  --dtype bfloat16 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --tensor-parallel-size 4 \
  --port 8000
```

---

## 生成流程（01_generate.py）

生成由单一入口 `01_generate.py` 完成，内部三阶段并行推进（逻辑在 `src/generation.py`）：
描述 → 出题 → 答案，结束后自动跑一次只读体检报告。下面分阶段说明每一步在做什么、产出什么。

```bash
# 基础用法：默认 LEVIR-CD+ 500 + SECOND 500，每图 2-5 问
python 01_generate.py

# 小规模冒烟：每个数据集各抽 2 张
python 01_generate.py --samples-per-dataset 2

# 全局混合随机采样 + 纳入 EBD
python 01_generate.py --datasets "EBD,LEVIR-CD+,SECOND" --num-images 50

# 跳过结束后的自动体检
python 01_generate.py --no-health-check
```

> 三个阶段不再是独立脚本，无法单独运行；参数全部挂在 `01_generate.py` 上（见 [参数详解](#参数详解)）。

### Phase 1: 生成受控图像描述（内部）

让教师模型先对每张图像生成保守、客观的描述，作为后续问答的"证据边界"。

**输出格式** (`outputs/01_descriptions.jsonl`):
```json
{
  "image_path": "/path/to/image.png",
  "dataset": "LEVIR-CD+",
  "description": "1. 场景类型：城市建成区...",
  "generated_at": "2025-01-23T10:30:00"
}
```

### Phase 2: 生成多样化问题（内部）

基于图像描述生成 7 类 VQA 问题（existence / location / spatial_relation / attribute / counting / unanswerable / ambiguous）。每张图按**软配额**（`src/sampling.py`）分配一份"建议类型清单"注入 prompt，使全局类型比例贴近目标、不易塌缩；模型对不适合的图可跳过某类型。

**输出格式** (`outputs/02_questions.jsonl`):
```json
{
  "image_path": "/path/to/image.png",
  "dataset": "LEVIR-CD+",
  "description": "...",
  "questions": [
    {"question_type": "existence", "question": "图像中是否有水体？"},
    {"question_type": "location", "question": "道路位于图像哪个区域？"}
  ],
  "generated_at": "2025-01-23T10:30:00"
}
```

### Phase 3: 生成带分析过程的答案（内部）

生成最终的 SFT 训练数据。两种形态按 `--direct-ratio` 混合：约 70% 带 `<analysis>`（步数自适应 1-3 步），约 30% 直答只有 `<answer>`。prompt 明确**以图像为准、描述仅供参考**。

**输出格式** (`outputs/03_answers_sft.jsonl`):
```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\n图像中是否有水体？"
    },
    {
      "role": "assistant",
      "content": "<analysis>\n1. 定位：...\n2. 观察：...\n3. 判断：...\n</analysis>\n<answer>...</answer>"
    }
  ],
  "images": ["/path/to/image.png"],
  "meta": {
    "task_type": "vqa_with_analysis",   // 直答样本为 "vqa_direct"
    "answer_mode": "analysis",            // analysis | direct
    "question_type": "existence",
    "source_dataset": "LEVIR-CD+",
    "auto_validation": {
      "is_valid": true,
      "warnings": [],
      "label": "pass"
    }
  }
}
```

> 直答样本的 `assistant.content` 只有 `<answer>…</answer>`，没有 `<analysis>`；`meta.answer_mode` 标记每条用的是哪种模式。

---

## 参数详解

参数全部挂在 `01_generate.py` 上（完整见 `python 01_generate.py -h`）。

### 模型 / 连接

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 自动探测 | 模型名称，如 `Qwen/Qwen3-VL-32B-Instruct` |
| `--base-url` | `http://10.129.107.145:8001/v1` | vLLM 服务地址 |
| `--retries` | `3` | API 调用失败后的最大重试次数 |

### 采样

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-root` | `dataset` | 数据集根目录 |
| `--datasets` | `LEVIR-CD+,SECOND` | 要处理的数据集，逗号分隔 |
| `--num-images` | `None` | 全局混合随机采样总数；传入后优先于 `--samples-per-dataset` |
| `--samples-per-dataset` | `LEVIR-CD+=500,SECOND=500` | 每数据集采样数；可写单一整数对所有数据集生效 |
| `--ebd-per-disaster` | `None` | EBD 按 7 种灾害子目录各抽 N 张 |
| `--seed` | `42` | 采样 + 软配额 + analysis/direct 分配的随机种子 |
| `--manifest` / `--refresh-manifest` | — | 复用 / 强制重抽样本清单 |

### 出题 / 答案

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-questions` | `5` | 每张图问题数上限（软配额在 [min, num] 区间分配类型） |
| `--min-questions` | `2` | 每张图问题数下限 |
| `--direct-ratio` | `0.3` | 直答（无 analysis）样本占比，打散固定起手式 |

### 运行 / 输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-concurrency` | `24` | 三阶段共用的最大并发请求数 |
| `--output-dir` | `outputs` | 输出目录（描述/问题/答案/stats/manifest 都落在这里） |
| `--no-health-check` | 关 | 跳过生成结束后的自动体检 |

---

## 常见问题排查

### Q: 连接vLLM失败？

**检查服务是否启动：**
```bash
curl http://10.129.107.145:8001/v1/models
```

**确认端口正确：**
```bash
# 如果服务在8000端口
python 01_generate.py --base-url http://localhost:8000/v1
```

### Q: 模型名称不匹配？

**查看vLLM实际加载的模型：**
```bash
curl -s http://10.129.107.145:8001/v1/models | python -m json.tool
```

然后用 `--model` 参数指定正确的模型名。

### Q: 断点续传不生效？

断点续传依赖 **图像绝对路径** 作为唯一标识。如果移动了数据集目录，旧的已处理记录会失效。

解决：删除输出文件重新运行，或手动过滤已处理的。

### Q: 生成的answer格式不正确？

自动质检会标记格式问题，在输出统计中可以看到：
- `format_fail`: 缺少必要的标签
- `warning`: 包含高风险词或其他警告

可以查看 `meta.auto_validation.warnings` 了解具体原因。

### Q: 如何调整生成的temperature？

编辑 `src/prompts.py` 或直接修改对应脚本中的 `temperature` 参数：
- 描述生成: 0.3 (保守)
- 问题生成: 0.4 (适度多样化)
- 答案生成: 0.3 (稳定格式)

---

## 高级用法

### 1. 分数据集并行处理

用不同的 `--output-dir` 把各数据集隔离到独立目录，互不干扰、可同时跑：

```bash
# 终端1：处理EBD数据集
python 01_generate.py --datasets EBD --num-images 50 --output-dir outputs/ebd

# 终端2：处理LEVIR-CD+数据集
python 01_generate.py --datasets LEVIR-CD+ --num-images 50 --output-dir outputs/levir
```

### 2. 自定义Prompt模板

编辑 `src/prompts.py` 中的模板：

```python
# 修改描述生成的要求
DESCRIPTION_PROMPT = """
你是一位专业的遥感图像解译人员...
[你的自定义内容]
"""
```

### 3. 添加新的数据集支持

编辑 `src/scanner.py` 中的 `DATASET_CONFIGS`:

```python
DATASET_CONFIGS = {
    "MY_DATASET": {
        "glob_patterns": ["**/images/*.png"],
        "image_filter": lambda p: True,  # 不过滤
    },
    # ...
}
```

### 4. 调整自动质检规则

编辑 `src/validator.py` 中的验证逻辑：
- 添加/移除高风险词
- 调整 analysis 步数范围（现为自适应 1-3 步）或 `expect_analysis` 双模式校验
- 添加新的问题类型检查（如 unanswerable / ambiguous 的拒答表达校验）

### 5. 快速检查数据质量

```bash
# 统计自动质检通过率
python -c "
import json
pass_count = total = 0
with open('outputs/03_answers_sft.jsonl') as f:
    for line in f:
        r = json.loads(line)
        total += 1
        if r['meta']['auto_validation']['label'] == 'pass':
            pass_count += 1
print(f'Pass: {pass_count}/{total} ({pass_count/total*100:.1f}%)')
"
```

---

## 完整运行流程（从零到训练集）

整条链路：**起 vLLM → 生成数据（自动体检）→ 人工抽检 → 转换 + 降采样 → SFT 训练**，
对应 4 个入口 `01~04`。下面命令可直接复制。

> 前提：图像数据集在 `--dataset-root`（默认 `dataset/`）下；把 `http://你的服务器:8001` 换成你的 vLLM 地址。

```bash
# ======== Step 0. 起教师模型（vLLM，OpenAI 兼容口）。详见“启动vLLM服务”节 ========
CUDA_VISIBLE_DEVICES=1,2 \
vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --port 8001

# ======== Step 1. 生成数据（三阶段并行）。结束后自动跑一次体检 ========
# --num-images：本次抽多少张图（不是样本数）。
python 01_generate.py \
  --dataset-root dataset \
  --datasets "LEVIR-CD+,SECOND" \
  --num-images 1000 \
  --num-questions 5 --min-questions 2 \
  --direct-ratio 0.3 \
  --max-concurrency 24 \
  --base-url http://你的服务器:8001/v1
# 产出：outputs/01_descriptions.jsonl / 02_questions.jsonl / 03_answers_sft.jsonl
#       outputs/sample_manifest.jsonl（本次抽样清单，复现/续跑用）

# ======== Step 2. 人工抽检（浏览器）。发布前必经一步 ========
python 02_review.py                # 打开 http://127.0.0.1:8008，逐条通过/不通过
# 审批结果写入 outputs/manual_reviews.jsonl

# ======== Step 3. 转 Unsloth 训练集 + 过滤 + 降采样（在 Linux 训练机上跑） ========
# --image-root 改成训练机上 dataset 的真实绝对路径；--rebalance 按配比降采样
python 03_convert.py \
  --image-root /your/abs/path/to/dataset \
  --rebalance \
  --check-images
# 产出 outputs/sft/train.jsonl 与 val.jsonl

# ======== Step 4. SFT 训练（Unsloth Core，Qwen3-VL + LoRA，仅 Linux+GPU） ========
# 04_train.py 是扁平脚本（无命令行参数）：超参直接改脚本顶部各 section 的常量
python 04_train.py
```

> 训练细节、参数与数据格式说明见下文 [SFT 训练（Unsloth / Qwen3-VL）](#sft-训练unsloth--qwen3-vl)。

**几个要点：**

- **`--num-images` 是"抽几张图"，不是"生成几条样本"。** 每张图出 2-5 个问题、每问 1 答，所以样本数 ≈ 图数 × 3 上下。
- Step 1 跑完会**自动体检**（只读，打印配比/塌缩/拒答率/编码报告 + 红线告警），不想要就加 `--no-health-check`。
- **降采样（`03_convert.py --rebalance`）建议在人工抽检之后再开**：它会按配比剔样本（主要压 counting），应有意识地在质检闸之后用，不要图省事提前降。
- 想单独再看体检报告：`python -c "from src import health; health.report(health.load('outputs/03_answers_sft.jsonl'))"`

### 默认采样规模

| 数据集 | 采样数量 | 说明 |
|--------|---------|------|
| LEVIR-CD+ | 500 | 城市建筑与道路场景 |
| SECOND | 500 | 多类地表覆盖场景 |
| **合计** | **1000 张图像** | 每图 2-5 个问题，预计 **2000–5000 个 VQA 样本** |

> 默认不启用 EBD（灾后场景）。若需要纳入 EBD，可显式传 `--datasets "LEVIR-CD+,SECOND,EBD" --samples-per-dataset "LEVIR-CD+=500,SECOND=500,EBD=140" --ebd-per-disaster 20`，按 7 类灾害子目录各抽 20 张。

#### 问题数量自适应

`--num-questions 5 --min-questions 2` 表示每张图生成 **2-5 个**问题，模型会根据图像内容自适应：

- 图像内容简单（如纯水体、单一植被）→ 2 个问题即可
- 图像内容丰富（多类地物、空间关系复杂）→ 最多 5 个问题
- 解析后若超过 5 个会硬截断到 5 个，不会让模型为了凑数而重复

#### 行为说明

- Phase 1 一旦产出某张图的描述，就立刻把这张图交给 Phase 2
- Phase 1 和 Phase 2 会并行推进
- Phase 3 会等到 Phase 1 和 Phase 2 都全部完成后，再统一开始
- 三个阶段共用一个并发上限，默认最多同时跑 24 个请求
- 首次运行会把采样结果固化到 `outputs/sample_manifest.jsonl`
- 后续默认复用同一份 manifest 做断点续跑，不会重新随机
- 如果确实要重新抽样，显式传 `--refresh-manifest`
- 三阶段全部完成后，会**自动跑一次只读数据体检**并打印报告（`--no-health-check` 可关）

完成数据生成后，进入 [人工审批前端](#人工审批前端) 完成抽样人工质检，再做 [数据体检与降采样](#数据体检与降采样)。

---

## 数据体检与降采样

发布前的数据治理逻辑在 `src/health.py`：体检（只读，给报告+红线告警）+ 按配比降采样。
体检由 `01_generate.py` 在生成结束后**自动调用**；降采样在 `03_convert.py --rebalance` 时触发。

```bash
# 体检（只读）：配比/归一化熵/问题前缀塌缩/答案模板化/拒答率/编码自检
python -c "from src import health; health.report(health.load('outputs/03_answers_sft.jsonl'))"

# 转换时按目标配比降采样（人工抽检后再开）：主要把 counting 压到 5%
python 03_convert.py --rebalance --image-root /your/abs/path/to/dataset
```

**红线判定**（`report()` 返回 True 表示触发；`01_generate.py` 会据此打印告警）：

| 维度 | 红线 |
|---|---|
| question_type 归一化熵 | < 0.7（类型分布失衡） |
| 问题前缀 Top-3 合计 | > 60%（Question Distribution Collapse） |
| answer / analysis 单一开头 | > 20%（过度模板化） |
| 无 direct 直答样本 | 全量带 analysis |

其余指标（拒答率是否在 5%-10%、各题型答案长度、编码乱码）仅提示，不卡红线。

---

## 人工审批前端

人工审批是 SFT 数据进入训练前的最后一步、也是必走一步。自动质检只能拦截格式与硬规则错误，对幻觉、错答、计数误差等只能依赖人工判定。仓库内置一个轻量本地审阅工具 `02_review.py`（逻辑在 `src/review.py`）来支撑这一步。

### 启动

```bash
python 02_review.py \
  --input outputs/03_answers_sft.jsonl \
  --reviews outputs/manual_reviews.jsonl \
  --port 8008
```

不传任何参数也可以，全用默认值：

```bash
python 02_review.py
```

启动后浏览器打开：

```text
http://127.0.0.1:8008
```

`Ctrl+C` 停止服务。

### 命令行参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--input` | `outputs/03_answers_sft.jsonl` | 要审核的 SFT JSONL，由 `01_generate.py` 产出 |
| `--reviews` | `outputs/manual_reviews.jsonl` | 审核结果落盘文件（不存在会自动创建） |
| `--host` | `127.0.0.1` | 监听地址，默认仅本机访问；要别的机器访问改 `0.0.0.0` |
| `--port` | `8008` | 监听端口 |

### 界面功能

- **左右分栏布局**：左侧大图，右侧问题 / 答案 / 自动质检 / 审批操作；不需要上下滚页。
- **审批操作**：单条样本进行 `通过 / 不通过` 判定，可写备注。
- **筛选**：`未审批 / 全部 / 仅通过 / 仅不通过` 四种过滤模式。
- **审批结果**：写入 `outputs/manual_reviews.jsonl`，**幂等覆盖**——同一样本多次审批以最后一次为准。
- **吸底操作栏**：长答案在右侧滚动时，"通过/不通过"按钮始终保持在视野内。

### 注意事项

- **样本主键**由 `image_path | question` 拼成。一旦图片路径或问题文本变更，旧的审批记录会"对不上"被视为未审。
- **新增样本要重启**：`02_review.py` 启动时一次性加载样本，运行期间不会自动重读 `--input` 文件，追加了新数据需要 Ctrl+C 后重启。
- **图片相对路径**：服务端按 JSONL 里 `images[0]` 的路径直接读图，相对路径相对于 **启动 server 时的当前目录**——建议在项目根目录启动。
- **没有鉴权**：默认绑 `127.0.0.1`，如果改 `--host 0.0.0.0` 请自行注意网络隔离。

### 下游消费

把 `outputs/manual_reviews.jsonl` 与 `outputs/03_answers_sft.jsonl` 按 `id`（即 `image_path|question`）join，仅保留 `decision == "approved"` 的样本，即可得到人工审核通过的 gold/silver 集合用于 SFT 训练。

---

## SFT 训练（Unsloth / Qwen3-VL）

发布集出来后，用 [Unsloth](https://unsloth.ai/docs) Core 对 Qwen3-VL 做视觉 LoRA 微调。
两个入口：`03_convert.py`（数据处理，Win/Linux 都能跑）+ `04_train.py`（训练，**仅 Linux+GPU**）。

### 为什么要 `03_convert.py` 这一步？

生成阶段的 `03_answers_sft.jsonl` 是**给人看 / 给审阅前端用**的格式，Unsloth 不能直接吃。差异有三：

1. **图像靠 `<image>` 文本占位符**，而 Unsloth 要的是结构化 `content` 列表里的图像对象（见下文格式）。
2. **图片路径是生成时写死的 `/home/charles/mycode/sft+rl/dataset/...`**，训练机上根目录多半不一样，要重映射。
3. 需要**过滤掉未通过质检的样本**，并切出 train/val。

`03_convert.py` 就是把这三件事一次做掉（逻辑在 `src/convert.py`），产出 `04_train.py` 能直接读的 `train.jsonl` / `val.jsonl`。

### Step A：装训练依赖（Linux）

```bash
# 先按 Unsloth 官方指引装好匹配的 torch+CUDA，再装本仓库训练依赖
pip install -r requirements-train.txt

# 训练用 SwanLab 做可视化（loss/学习率/显存曲线）。首次登录一次即可：
swanlab login            # 贴上 swanlab.cn 的 API key；或设环境变量 SWANLAB_API_KEY
```

> 训练时 `04_train.py` 已接入 `SwanLabCallback`，会自动把曲线推到看板（项目/实验名见脚本顶部 `SWANLAB_PROJECT` / `SWANLAB_EXPERIMENT` 常量）。想完全离线只在本机看，给回调加 `mode="local"`。

### Step B：转训练集

```bash
python 03_convert.py \
  --image-root /your/abs/path/to/dataset \  # 训练机上 dataset 的真实绝对路径，必填
  --val-ratio 0.02 \
  --rebalance \                            # 可选：按配比降采样，主要压 counting
  --check-images                           # 图片在本机时建议开，校验每张图都能打开
# 产出 outputs/sft/train.jsonl 与 val.jsonl
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--src` | `outputs/03_answers_sft.jsonl` | 输入答案 jsonl |
| `--out-dir` | `outputs/sft` | 输出目录 |
| `--old-root` | `/home/charles/mycode/sft+rl/dataset` | 被替换的原图片根目录 |
| `--image-root` | 空 | **训练机上的图片根目录**；留空则不改路径（到 Linux 上务必传） |
| `--keep-warning` | 关 | 也保留 `warning` 样本（默认只留 `pass`，永远剔除 `format_fail`） |
| `--rebalance` | 关 | 转换前按目标配比降采样（主要把 counting 压到 5%） |
| `--check-images` | 关 | 校验每张图是否存在（需图片在本机） |
| `--val-ratio` | `0.02` | 验证集比例 |

> 路径重映射**始终输出 POSIX 斜杠**，即便在 Windows 上跑 `03_convert.py`，产出的路径到 Linux 也直接可用。

### Step C：训练

`04_train.py` 是**扁平脚本**（照搬 Unsloth 官方 Qwen3-VL Vision notebook 的结构，刻意不封装、不加命令行参数），方便逐段对照学习。**改超参就直接改脚本顶部各 section 里的常量。**

```bash
# 冒烟：把脚本里 `# max_steps = 30` 那行取消注释（覆盖 epochs，只跑 30 步），确认数据/显存/模板都通
python 04_train.py

# 正式训练：把 max_steps 注释回去，按需调 num_train_epochs 等常量，再跑
python 04_train.py
```

脚本分六步，每个参数都在所属步骤里就近注释：

| 步骤 | 关键常量 | 说明 |
|------|---------|------|
| ① 加载模型 | `MODEL_NAME` / `load_in_4bit` | 基座（默认官方 4bit 量化版）；显存够可关 4bit |
| ② 挂 LoRA | `r` / `lora_alpha` / `finetune_vision_layers` | LoRA 秩=容量；视觉层建议保持训 |
| ③ 准备数据 | `convert_to_conversation` | 把每行 `{image,question,answer}` 转成 Unsloth 对话 + `Image.open` 读图 |
| ④ 训练器 | `SFTConfig(...)` | 批大小/时长/学习率/精度（见下「起步配方」） |
| ⑤ 训练 | — | `trainer.train()` |
| ⑥ 保存 | `OUTPUT_DIR` | 存 LoRA；末尾有合并 16bit 的可选注释行 |

要点：
- 用 `FastVisionModel` + `UnslothVisionDataCollator`，**只训练 LoRA 适配器**，基座冻结。
- 数据是 eager 列表（一次读进内存），几千张没问题；规模很大时改 `with_transform` 惰性读图（脚本里有注释）。
- 输出 LoRA 适配器到 `OUTPUT_DIR`；要合并 16bit 权重就取消脚本末尾那行注释。

### 起步配方（~3795 条规模）

当前数据约 3720 条训练样本，**有效 batch = `per_device_train_batch_size` × `gradient_accumulation_steps` = 16**，每 epoch ≈ 3720 / 16 ≈ **232 步**。脚本里 `SFTConfig` 的默认就是下面这套，按需在脚本顶部微调：

| 常量（在 `SFTConfig` 内） | 默认 | 调参取舍 |
|------|------|------|
| `num_train_epochs` | `2` | 3720 条偏小，从 2 起看 loss 再加到 3；>3 易过拟合到模板腔 |
| `per_device_train_batch_size` / `gradient_accumulation_steps` | `2` / `8` | 有效 batch 16；OOM 就 batch 改 1、accum 改 16，乘积不变 |
| `learning_rate` | `2e-4` | LoRA 标准值；全参才用 1e-5 量级，LoRA 用错几乎不学 |
| `r` / `lora_alpha`（在 `get_peft_model`） | `16` / `16` | alpha=r；几千条够用，上到几万或欠拟合再升 32 |
| `finetune_vision_layers`（在 `get_peft_model`） | `True` | 遥感图与自然图差异大，保持训视觉塔；OOM 才退一步关掉 |
| `load_in_4bit`（在 `from_pretrained`） | `True` | 显存紧保持 4bit；A100 80G 可设 False |

> 想监控过拟合，可在 `SFTConfig` 里加 `eval_strategy="steps"` + `eval_steps=100` 并给 `SFTTrainer` 传一个 `eval_dataset`（同样用 `convert_to_conversation` 转 `outputs/sft/val.jsonl`）。基础版照搬 notebook 没带验证集，保持最简。

### Unsloth 的数据格式 vs 我们的生成格式

**Unsloth 视觉微调要的格式**（`04_train.py` 在内存里构造，不用你手写）：

```python
[
  {"role": "user", "content": [
      {"type": "image", "text": None, "image": <PIL.Image 对象>},
      {"type": "text",  "text": "图像中是否有水体？"}]},
  {"role": "assistant", "content": [
      {"type": "text", "text": "<analysis>...</analysis>\n<answer>...</answer>"}]},
]
```

关键点：`content` 是一个**列表**，图像和文本是并列的结构化元素，图像是**真正的图对象**（或可被 processor 解析的对象），不是字符串占位符。

**我们生成的格式**（`03_answers_sft.jsonl`）：

```json
{"messages": [
   {"role": "user", "content": "<image>\n图像中是否有水体？"},
   {"role": "assistant", "content": "<analysis>...</analysis>\n<answer>...</answer>"}],
 "images": ["/home/charles/.../05796.png"]}
```

**为什么不能直接喂给 Unsloth：**

| 维度 | 我们的格式 | Unsloth 要的 | 后果 |
|------|-----------|-------------|------|
| 图像位置 | `content` 里的 `<image>` 文本占位符 + 独立 `images` 字段 | `content` 列表里的结构化图像元素 | 直接读会把 `<image>` 当普通文本，图像喂不进去 |
| `content` 类型 | 字符串 | 列表（text/image 元素） | collator 解析不到图像 token |
| 图片引用 | 文件路径字符串 | PIL/可解析图对象 | 需要在 `03_convert.py`/`04_train.py` 里 `Image.open` 加载 |
| 路径根目录 | 生成机的 `/home/charles/...` | 训练机真实路径 | 不重映射会找不到图 |

> 说明：我们这套格式是 **LLaMA-Factory / ShareGPT 风格**（`messages` + 平行 `images` + `<image>` 占位符），它本身是业界常见的 VLM SFT 格式，**LLaMA-Factory 能直接读**。只是 **Unsloth 用的是另一套（结构化 content）**，所以中间需要 `03_convert.py` 做一次转换。换句话说不是你的数据"错了"，是两个训练框架的输入约定不同。
