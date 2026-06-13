# 详细使用指南

## 目录
1. [模型配置](#模型配置)
2. [三阶段运行流程](#三阶段运行流程)
3. [参数详解](#参数详解)
4. [常见问题排查](#常见问题排查)
5. [高级用法](#高级用法)
6. [人工审批前端](#人工审批前端)

---

## 模型配置

### 方式零：自动探测（默认行为 ✨）

脚本启动时会自动调用 vLLM 的 `/v1/models` 接口，探测服务端实际加载的模型名称：

```bash
# 不需要指定 --model，自动探测
python 01_generate_desc.py
```

运行时会显示：
```
[自动探测] 服务端模型: Qwen/Qwen3-VL-32B-Instruct
```

这是**最推荐**的方式，避免模型名称不匹配问题。

---

### 方式一：通过命令行参数指定

所有三个脚本都支持 `--model` 和 `--base-url` 参数：

```bash
python 01_generate_desc.py \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --base-url http://10.129.107.145:8001/v1
```

### 方式二：通过环境变量设置

```bash
# 设置环境变量
export VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct

# 脚本会自动读取
python 01_generate_desc.py
```

### 模型名称优先级

```
用户指定 --model > 环境变量 VLLM_MODEL > 自动探测 > 默认值
```

---

## 启动vLLM服务

### Qwen3-VL 推荐配置

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --port 8001
```

### 使用Swift部署（备选）

```bash
swift deploy --model Qwen/Qwen3-VL-32B-Instruct \
  --port 8001 \
  --infer-backend vllm
```

### 多卡部署示例

```bash
# 4卡部署
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --port 8000
```

---

## 三阶段运行流程

### Phase 1: 生成受控图像描述

**作用**：让教师模型先对每张图像生成保守、客观的描述，作为后续问答的"证据边界"。

```bash
# 基础用法：默认只跑 LEVIR-CD+ 和 SECOND，各采样 10 张
python 01_generate_desc.py

# 指定数据集并设置每个数据集采样数
python 01_generate_desc.py \
  --datasets "LEVIR-CD+,SECOND" \
  --samples-per-dataset 10 \
  --seed 12345

# 如果需要恢复全局混合随机采样
python 01_generate_desc.py \
  --datasets "EBD,LEVIR-CD+,SECOND" \
  --num-images 50

# 自定义输出路径
python 01_generate_desc.py \
  --output outputs/my_descriptions.jsonl
```

**输出格式** (`outputs/01_descriptions.jsonl`):
```json
{
  "image_path": "/path/to/image.png",
  "dataset": "LEVIR-CD+",
  "description": "1. 场景类型：城市建成区...",
  "generated_at": "2025-01-23T10:30:00"
}
```

---

### Phase 2: 生成多样化问题

**作用**：基于图像描述生成8类VQA问题，覆盖存在性、属性、位置、空间关系等。

```bash
# 基础用法（使用默认输入）
python 02_generate_questions.py

# 自定义问题数量
python 02_generate_questions.py \
  --num-questions 10 \
  --min-questions 6

# 指定输入输出
python 02_generate_questions.py \
  --input outputs/my_descriptions.jsonl \
  --output outputs/my_questions.jsonl
```

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

---

### Phase 3: 生成带分析过程的答案

**作用**：生成最终的SFT训练数据，包含结构化分析过程和简洁答案。

```bash
# 基础用法
python 03_generate_answers.py

# 指定输入输出
python 03_generate_answers.py \
  --input outputs/my_questions.jsonl \
  --output outputs/my_final_sft.jsonl
```

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
    "task_type": "vqa_with_analysis",
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

---

## 参数详解

### 通用参数（所有脚本都支持）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 自动探测 | 模型名称，如 `Qwen/Qwen3-VL-32B-Instruct` |
| `--base-url` | `http://10.129.107.145:8001/v1` | vLLM服务地址 |
| `--retries` | `3` | API调用失败后的最大重试次数 |

### Phase 1 特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-root` | `dataset` | 数据集根目录 |
| `--datasets` | `LEVIR-CD+,SECOND` | 要处理的数据集，逗号分隔 |
| `--num-images` | `None` | 全局混合随机采样总数 |
| `--samples-per-dataset` | `10` | 每个数据集采样数量 |
| `--seed` | `42` | 随机种子，用于复现采样结果 |
| `--output` | `outputs/01_descriptions.jsonl` | 输出文件路径 |

### Phase 2 特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `outputs/01_descriptions.jsonl` | Phase 1输出文件 |
| `--output` | `outputs/02_questions.jsonl` | 输出文件路径 |
| `--num-questions` | `8` | 每张图尝试生成的问题数量 |
| `--min-questions` | `5` | 每张图最少需要的问题数量 |

### Phase 3 特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `outputs/02_questions.jsonl` | Phase 2输出文件 |
| `--output` | `outputs/03_answers_sft.jsonl` | 最终SFT输出文件 |

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
python 01_generate_desc.py --base-url http://localhost:8000/v1
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

```bash
# 终端1：处理EBD数据集
python 01_generate_desc.py --datasets EBD --num-images 50 \
  --output outputs/ebd_descriptions.jsonl

# 终端2：处理LEVIR-CD+数据集
python 01_generate_desc.py --datasets LEVIR-CD+ --num-images 50 \
  --output outputs/levir_descriptions.jsonl
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
- 调整analysis步骤数要求
- 添加新的问题类型检查

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

## 完整运行示例

```bash
# ==================== 完整流程 ====================

# 1. 启动vLLM服务
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size 2 --port 8001 &

# 2. Phase 1: 生成200张图像的描述
python 01_generate_desc.py \
  --num-images 200 \
  --seed 42

# 3. Phase 2: 每张图生成8个问题
python 02_generate_questions.py \
  --num-questions 8

# 4. Phase 3: 生成带分析的答案
python 03_generate_answers.py

# 5. 查看最终结果
ls -lh outputs/
```

### 流水线并行运行

如果希望 Phase 1/2/3 同时推进，而不是等上一阶段全部结束后再开始下一阶段，可以使用新的流水线并行入口：

```bash
python run_parallel_pipeline.py \
  --datasets "LEVIR-CD+,SECOND" \
  --samples-per-dataset 10 \
  --max-concurrency 24
```

它的行为是：
- Phase 1 一旦产出某张图的描述，就立刻把这张图交给 Phase 2
- Phase 1 和 Phase 2 会并行推进
- Phase 3 会等到 Phase 1 和 Phase 2 都全部完成后，再统一开始
- 三个阶段共用一个并发上限，默认最多同时跑 24 个请求
- 首次运行会把采样结果固化到 `outputs/sample_manifest.jsonl`
- 后续默认复用同一份 manifest 做断点续跑，不会重新随机
- 如果确实要重新抽样，显式传 `--refresh-manifest`

完成数据生成后，进入 [人工审批前端](#人工审批前端) 完成抽样人工质检。

---

## 人工审批前端

人工审批是 SFT 数据进入训练前的最后一步、也是必走一步。自动质检只能拦截格式与硬规则错误，对幻觉、错答、计数误差等只能依赖人工判定。仓库内置一个轻量本地审阅工具 `review_server.py` 来支撑这一步。

### 启动

```bash
python review_server.py \
  --input outputs/03_answers_sft.jsonl \
  --reviews outputs/manual_reviews.jsonl \
  --port 8008
```

不传任何参数也可以，全用默认值：

```bash
python review_server.py
```

启动后浏览器打开：

```text
http://127.0.0.1:8008
```

`Ctrl+C` 停止服务。

### 命令行参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--input` | `outputs/03_answers_sft.jsonl` | 要审核的 SFT JSONL，由 `03_generate_answers.py` 产出 |
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
- **新增样本要重启**：`review_server.py` 启动时一次性加载样本，运行期间不会自动重读 `--input` 文件，追加了新数据需要 Ctrl+C 后重启。
- **图片相对路径**：服务端按 JSONL 里 `images[0]` 的路径直接读图，相对路径相对于 **启动 server 时的当前目录**——建议在项目根目录启动。
- **没有鉴权**：默认绑 `127.0.0.1`，如果改 `--host 0.0.0.0` 请自行注意网络隔离。

### 下游消费

把 `outputs/manual_reviews.jsonl` 与 `outputs/03_answers_sft.jsonl` 按 `id`（即 `image_path|question`）join，仅保留 `decision == "approved"` 的样本，即可得到人工审核通过的 gold/silver 集合用于 SFT 训练。
