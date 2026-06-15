# 纯蒸馏的遥感 VQA SFT 数据构建

本项目当前聚焦于构建**单图遥感 VQA SFT 冷启动数据**。采用**纯蒸馏**方式：把"图像 + 问题"直接交给更强的教师模型（Qwen3.5-27B），它怎么回答就原样收作训练标签，**不强加任何输出格式**，只保留一条护栏——看不清就说不确定。

核心目标不是生成长篇遥感解译报告，而是为后续 GRPO 提供一个足够稳定的遥感 VQA 初始模型：模型先学会看遥感图、理解问题、用自然口吻回答，再通过更高质量的 GRPO 数据继续提升识别准确率。

> **为什么是纯蒸馏（而非早期的固定格式）？** 早期版本强制每条答案套 `<analysis>…</analysis><answer>…</answer>` 模板。实践发现：在 ~3700 条又窄又同构的遥感数据上，固定格式会让 2B 学生模型把"无论问什么都输出这套模板腔"当成最省力策略，**严重放大灾难性遗忘**（通用问答/自然图理解被模板覆盖）。改成纯蒸馏后，学生学的是 27B 教师的自然分布，从源头上拆掉了格式坍缩这个遗忘主因。代价是丢掉了答案的机器可解析结构（若后续 RL 需要抽取最终答案，再另行约定）。

## 改进思路：从"能生成数据"到"设计数据分布"

这套流程不是一开始就长这样，而是针对蒸馏类 SFT 数据的几个典型陷阱逐步演进而来。记录这条思路，是为了说明每个设计决策背后要规避的具体风险。

| 维度 | 一开始的做法 | 暴露的问题 | 改进后的做法 |
|------|------------|-----------|------------|
| **答案格式** | 每条都强制 `<analysis>` 固定 3 步（定位/观察/判断）+ `<answer>`；后改为步数自适应 + 混 30% 直答 | 即便放宽，**固定 XML 脚手架每条都出现**仍把窄数据上的 2B 学生拽向单一模板腔，放大灾难性遗忘 | **纯蒸馏**：撤掉一切格式约束与格式校验，图+问题直接问教师、原样收下，只留"看不清说不确定"一条护栏 |
| **描述的角色** | Phase3 答案 = `图 + 描述 + 问题`，描述是强依赖 | 描述（Phase1）一旦出错，问题和答案跟着错，形成"错描述→错问答"的自洽闭环 | 改为"以图像为准、描述仅弱参考、冲突以图为准"，并在 Phase2/3 都加该约束 |
| **问题类型配比** | 让模型自由出题 | counting 实测占到 ~20%；遥感计数噪声极高（遮挡/分辨率/边界模糊） | 软配额（`src/sampling.py`）按目标分布给每图分配建议类型；counting 压到 5% |
| **问题多样性** | 无约束 | 长期运行易塌缩成"是否有建筑/道路/植被"刷屏（Question Distribution Collapse） | prompt 显式要求同类不重复句式/对象 + 体检脚本监控问题前缀塌缩红线 |
| **拒答能力** | 训练集 100% 都有确定答案 | 模型学到"必须回答"，遇到看不出的就硬编 → VL 微调最易翻车点 | 新增 `unanswerable`（答不了）+ `ambiguous`（多解）两类拒答样本，合计 ~6% |
| **质量把关** | 仅生成时的格式/硬规则自动质检 | 模板化、配比失衡、分布塌缩这些"软问题"无人监控 | 发布前数据体检（配比/归一化熵/前缀塌缩/模板化红线，可接 CI）+ 按配比降采样 |

一句话概括：早期解决的是"**能不能稳定生成带证据链的数据**"，本轮解决的是"**生成的数据分布是否健康**"——对 Qwen3-VL 这类 VLM 的 SFT，后者往往比 prompt 写得漂不漂亮更影响最终模型行为。

> 仍待处理（规模上来后）：当前三阶段默认同一教师模型，问答风格同源；50k+ 时建议各阶段换不同教师（`--base-url`/`--model`）以提升监督多样性。

## 项目结构

```text
sft_rl/
├── 01_generate.py              # 入口①：生成基础数据 + 自动质量分析报告
├── 02_review.py                # 入口②：人工审批前端
├── 03_convert.py               # 入口③：转 Unsloth 格式（纯蒸馏下默认全保留）+ 切 train/val
├── 04_train.py                 # 入口④：Qwen3-VL 视觉 SFT（扁平脚本，照搬 Unsloth notebook，仅 Linux+GPU）
│
├── src/                        # 核心功能模块（01~03 入口是薄壳，逻辑在这里；04 刻意扁平不入 src）
│   ├── scanner.py             # 图像扫描与分层采样
│   ├── vlm_client.py          # VLM 客户端封装（带重试）
│   ├── validator.py           # 旧格式自动质检规则（纯蒸馏后 generation 不再调用，保留备查）
│   ├── sampling.py            # 问题类型软配额调度（控制分布）
│   ├── prompts.py             # 所有 Prompt 模板集中管理
│   ├── generation.py          # 三阶段并行生成引擎（描述→出题→答案）
│   ├── health.py              # 数据体检 + 按配比降采样
│   ├── convert.py             # 转 Unsloth 训练集 + 质检过滤 + 路径重映射
│   └── review.py             # 本地人工审批服务
│
├── scratch/learn/              # 学习/治理脚本
│   └── 01_inspect_template.py  # 查看 chat template 渲染结果
│
├── review/ui/                  # 人工审批前端静态资源（被 02_review.py 使用）
│
├── outputs/                    # 各阶段输出
│   ├── 01_descriptions.jsonl   # 图像描述结果
│   ├── 02_questions.jsonl      # 问题生成结果
│   ├── 03_answers_sft.jsonl    # 候选 SFT 数据
│   ├── sample_manifest.jsonl   # 本次抽样清单（可复现/续跑）
│   └── sft/                     # 03_convert.py 产出的 train.jsonl / val.jsonl
│
├── README.md / RUN_GUIDE.md    # 设计说明 / 运行指南
├── requirements.txt            # 数据生成依赖
└── requirements-train.txt      # 训练依赖（仅 Linux+GPU）
```

### 四个入口、一条主线

主目录只有 4 个入口脚本，按数据流先后排列；所有可复用逻辑都在 `src/`：

| 入口 | 职责 | 输入 → 输出 |
|------|------|------|
| `01_generate.py` | 三阶段生成 + 自动体检报告 | 遥感图像 → `outputs/03_answers_sft.jsonl` |
| `02_review.py` | 人工审批（发布前必经） | SFT 数据 → `outputs/manual_reviews.jsonl` |
| `03_convert.py` | 转 Unsloth 格式 + 切 train/val（纯蒸馏下默认全保留） | 答案 jsonl → `outputs/sft/train.jsonl`·`val.jsonl` |
| `04_train.py` | Qwen3-VL 视觉 SFT（LoRA） | 训练集 → LoRA 适配器 |

> 生成阶段内部仍是三阶段（描述→出题→答案）并行推进，逻辑在 `src/generation.py`；断点续传按图像绝对路径识别。详细使用说明见 `RUN_GUIDE.md`。

## 训练分工

推荐的整体训练路线：

```text
SFT:
  使用本项目蒸馏构建的遥感 VQA 数据（教师=Qwen3.5-27B）
  目标：学会看遥感图、理解 VQA 问题、用自然口吻基础识别与作答

GRPO:
  使用 GeoChat_Instruct 等更高质量遥感 instruction/reference 数据
  目标：进一步提升识别准确率和答案可靠性
```

SFT 阶段训练的是“会答、会用自然语言答、不瞎编”。GRPO 阶段再重点优化“答得更准”。

因此，SFT 数据可以容忍少量噪声，但不能形成系统性坏习惯。需要重点避免：

- 答案过长，变成长篇遥感报告。
- 编造图像中不可见的证据。
- 遇到不确定目标仍强行判断（护栏要求看不清就说不确定）。
- 单图问题回答行政建议、灾害成因或工程修复方案。
- 计数题编造过精确数字。
- 问题和答案不匹配。

推荐的 SFT 数据形态：

```text
短问题
+ 教师自然作答（不套固定格式）
+ 简洁、基于可见内容
+ 看不清就说不确定
+ 抽样人工审核
```

## 数据集

支持以下三个遥感数据集，需放置于 `DATASET_ROOT` 下：

| 数据集 | 使用图像 | 场景类型 | 目录结构 |
|--------|----------|----------|----------|
| **EBD** | `post_disaster` | 灾后遥感场景 | `EBD/{EVENT}/images/{EVENT}_{ID}_post_disaster.*` |
| **LEVIR-CD+** | `B` | 城市建筑与道路场景 | `LEVIR-CD+/{train\|test}/B/*.png` |
| **SECOND** | `im2` | 多类地表覆盖场景 | `SECOND/{train\|test}/im2/*.png` |

VQA 阶段只使用单张图像。对于双时相数据集，只取后时相或 post 侧图像，避免同一场景的 pre/post 内容重复进入单图 VQA。

## 数据目标

重构后的 SFT 数据训练目标是：

- 会识别遥感图像中的基础地物。
- 会理解短 VQA 问题。
- 会用自然语言简洁作答。
- 遇到不确定内容时会说明不确定，而不是强行编造。

最终样本采用 OpenAI/Swift 常见的多模态 SFT JSONL 格式（**纯蒸馏，无格式标签**）：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\n图像右侧道路附近是否分布有建筑？"
    },
    {
      "role": "assistant",
      "content": "是的。图像右侧那条浅灰色道路两侧排列着多个规则的矩形屋顶，属于沿道路分布的建筑。"
    }
  ],
  "images": ["/path/to/image.png"],
  "meta": {
    "task_type": "vqa_distill",
    "question_type": "existence",
    "source_dataset": "LEVIR-CD+"
  }
}
```

`assistant.content` 就是**教师模型对"图像+问题"的原始自然回答**——可能一句话直答，也可能带几句说明，完全由教师决定，本项目不再强制 `<analysis>`/`<answer>` 结构，也不再做格式校验。

> **为什么不套格式？** 早期版本强制（或半强制）每条都以「1. 定位 / 2. 观察 / 3. 判断」+ `<answer>` 起手。在 ~3700 条又窄又同构的数据上，固定脚手架会让 2B 学生把"无论问什么都套这套模板"当成最省力策略，**严重放大灾难性遗忘**——通用问答和自然图理解被模板腔覆盖。纯蒸馏让学生直接学 27B 教师的自然分布，从源头消除这个坍缩源。`meta.task_type` 统一为 `vqa_distill`（不再有 `answer_mode` 字段）。

> 代价：答案没有可机器解析的定界符。若后续 GRPO 需要从输出里抽取"最终答案"打分，到时再单独约定一个轻量定界（如只保留 `<answer>`），而不必回退到完整模板。

## 构建流程

SFT 数据建议分为三个生成阶段、一个自动质检阶段和一个抽样人工质检阶段。

### Phase 1: 受控图像描述

教师模型先对单张遥感图像生成受控描述，作为后续问题和答案生成的证据参考。

描述应覆盖：

- 场景类型：城市、乡村、农田、水体、植被、工业区、灾后区域等。
- 主要地物：建筑、道路、植被、水体、车辆、裸地、人工设施等。
- 空间关系：相邻、位于某方向、沿道路分布、被道路分隔等。
- 可计数对象：只统计清晰可见的对象，不强行计入遮挡或模糊目标。
- 不确定内容：无法确认的用途、类型或数量需要明确标记。

描述阶段只允许写图像中可见的内容，不输出行政建议、灾害成因、工程治理方案、精确坐标或精确面积。

### Phase 2: 问题生成

基于 `图像 + 受控描述` 生成问题，但 prompt 中明确**描述仅供参考、可能有误，以图像为准**。问题应覆盖多种 VQA 能力，而不是只做开放式描述。

问题类型与目标配比（`src/sampling.py:SCHEDULE_DIST`）：

| 类型 | 说明 | 目标占比 |
|------|------|----------|
| `existence` | 是否存在某类地物 | 33% |
| `location` | 某地物位于图像哪个区域 | 23% |
| `spatial_relation` | 两个地物之间的相对位置或邻接关系 | 19% |
| `attribute` | 颜色、形状、屋顶类型、材质或纹理 | 14% |
| `counting` | 清晰可见、数量较少、可准确点数的目标数量 | 5% |
| `unanswerable` | 仅凭该图像无法确认的问题（用途/材质/年代/权属等） | 3% |
| `ambiguous` | 图像存在多个合理解释、难以唯一确定（如"仓库还是厂房？"） | 3% |

> **配比调整说明**：相比早期版本特意**压低了 `counting`（15%→5%）**——遥感计数受遮挡、分辨率、边界模糊影响极大，强行数出"17 栋"往往实际是"15~20 栋"，噪声比 existence/location 高得多。同时新增两类拒答样本（合计约 6%）：`unanswerable`（答不了）与 `ambiguous`（多解）。若训练集 100% 都有确定答案，模型会学到"必须回答"，这是 VL 微调最容易翻车的点。

**配比如何"不差太远"——三层控制：**

1. **生成时软配额**（`src/sampling.py`）：为每张图按目标分布确定性地分配一份"建议类型清单"注入 prompt（如 `existence×2, location×1`）。单图独立、按 `(seed|image)` 哈希抽样，断点续跑/并行结果一致；图像足够多时聚合分布收敛到目标。**软约束**——prompt 允许模型对不适合的图（如无可数目标）跳过某类型，避免硬逼烂题。
2. **生成时反塌缩**：prompt 明确要求"同类问题不要用相同句式起手或问同一对象"，对抗 *Question Distribution Collapse*（"是否有建筑/道路/植被"刷屏）。
3. **发布时降采样**：兜底精确比例（见 [数据体检与降采样](#数据体检与降采样)）。

每张图生成 **2-5 个**问题（默认值，可通过 `--num-questions` / `--min-questions` 调整）。具体数量由模型根据图像内容自适应：内容简单时只出 2 个，内容丰富时最多 5 个；解析后超过上限会硬截断，避免为了凑数生成低质量问答。

### Phase 3: 纯蒸馏答案生成

把 `图像 + 问题` 直接交给教师模型作答，**怎么答就原样收下**。与早期最大的不同：

1. **不套任何格式**：去掉了 `<analysis>`/`<answer>` 模板、analysis/direct 分流、`--direct-ratio` 配比，以及生成时的格式校验（`validate_answer`/`salvage_answer`）。教师输出什么就是什么。
2. **只留一条护栏**：prompt（`ANSWER_DISTILL_PROMPT`）只要求"只依据图像可见内容回答；看不清就直接说不确定，不要猜测"。
3. **连描述都不传**：纯蒸馏直接让教师看图作答，不再把 Phase 1 描述作为参考文本喂进去——既更贴近"图+问"的真实输入，也不引入描述误差。

> 注意：教师答案不再经过格式/幻觉自动校验，质量完全取决于教师（Qwen3.5-27B）本身。系统性错误依赖 [Phase 5 人工抽检](#phase-5-抽样人工质检) 兜底。

每条产出 `meta.task_type = "vqa_distill"`，不再有 `answer_mode`、`auto_validation` 字段。

> **关于教师模型多样性（待办，规模上来后处理）**：当前三个阶段默认用同一教师模型，问题/答案的风格与偏好同源。脚本已支持各阶段独立传 `--base-url` / `--model`，数据量做到 50k+ 时建议**换不同教师**以提升监督信号多样性。这属于运行层配置，无需改代码。

### Phase 4: 轻量自动检查

> ⚠️ **纯蒸馏后，原来基于格式的强自动质检已大幅退场**：不再校验 `<analysis>`/`<answer>` 标签、步数、answer_mode 等。`03_convert.py` 因 `meta.auto_validation` 缺失而**默认全量保留**。质量把关的重心移到 Phase 5 人工抽检。

仍然有意义的最少检查：

- 是否是合法 JSONL。
- `messages` 和 `images` 字段是否完整。
- 图像路径是否存在。
- 答案非空、且大致回答了问题（看不清类是否给出了不确定表达）。
- 是否出现高风险幻觉词（精确坐标 / 精确面积 / 米级位移 / 灾害成因 / 行政处置 / 工程修复方案 / GNSS / InSAR 等）——这些超出单图 VQA 证据范围，出现后宜人工复核或剔除。

> `src/validator.py`、`src/health.py` 里那套面向旧格式的强校验/红线（analysis 开头塌缩、answer_mode 比例、"无 direct 直答"红线等）在纯蒸馏下已**部分失效**（如"无 direct 样本"会恒为真而误报），尚未清理。如需重新启用体检，得先按蒸馏形态改写这两个模块——目前作为已知技术债记录在此。

### Phase 5: 抽样人工质检

自动质检后，需要对候选 SFT 数据做分层抽样人工审核。人工质检不做全量，目标是估计每类数据的真实质量，并发现系统性错误。

推荐抽样维度：

- `source_dataset`：EBD / LEVIR-CD+ / SECOND。
- `question_type`：existence / attribute / location / spatial_relation / counting / unanswerable。

> 纯蒸馏后不再有 `answer_mode` / `auto_validation_label` 维度（这两个字段已移除）。

推荐抽样规模：

```text
每个 dataset × 每个 question_type 抽 20 条
```

如果某个桶样本较少，可以使用：

```text
min(20, 桶内样本数的 10%)
```

人工审核字段建议：

```text
sample_id
source_dataset
question_type
image_path
question
answer
image_supported
question_valid
answer_grounded
answer_correct
hallucination
severity
final_label
comment
```

人工判定标准：

- `gold`：图像支持问题，答案基于可见内容、正确简洁，无明显幻觉。
- `silver`：基本正确，存在轻微啰嗦或不精确，但不影响训练目标。
- `rejected`：问题不可答、答案错误、编造图像中不可见的内容、出现图像外推断或计数明显错误。

抽检结果应反向影响数据发布。如果某个 `dataset/question_type` 桶的人工通过率过低，该桶全量样本不应直接进入 SFT 主训练集，需要回退重生成、修改 prompt 或降低采样比例。

### 数据体检与降采样

> ⚠️ **纯蒸馏适配未完成**：`src/health.py` 现有的"答案侧"指标（answer/analysis 开头塌缩、answer_mode 比例、"无 direct 直答"红线）是为旧格式设计的，纯蒸馏下会失真甚至误报（如"无 direct 样本"恒触发）。**问题侧**指标（类型配比、归一化熵、问题前缀塌缩）仍然有效——这部分和答案格式无关。下文保留原描述，但答案侧请暂时忽略，待 health.py 按蒸馏形态改写后再启用。

体检逻辑在 `src/health.py`。`01_generate.py` 生成结束后会**自动调用它跑一次只读体检**（除非 `--no-health-check`），把"是否过度模板化、配比是否失衡"量化出来。发布前还可按目标配比降采样生成发布集（在 `03_convert.py` 里以 `--rebalance` 触发）。

**体检（只读，给报告 + 红线告警）：** 随 `01_generate.py` 自动产出；也可单独调用：

```bash
python -c "from src import health; health.report(health.load('outputs/03_answers_sft.jsonl'))"
```

报告分**问题侧**（分布塌缩，最危险）和**答案侧**（过度模板化）两部分：

问题侧：

- **question_type 配比 + 归一化熵**：熵越低越偏，< 0.7 判为类型分布失衡（红线）。
- **问题前缀 Top-N + 塌缩红线**：前 3 个前缀合计 > 60% 即判定 *Question Distribution Collapse*（红线）。这是 type 降采样治不了的问题——塌缩常发生在 existence 内部。

答案侧：

- **answer / analysis 开头 Top-N 分布**：任何单一开头占比超过阈值（默认 20%）即告警——过度模板化的直接信号。
- **answer_mode 比例**（analysis vs direct）是否接近 `--direct-ratio`。
- 各题型 **answer 长度** 分布。
- **拒答类占比**（unanswerable + ambiguous）是否落在 5%-10% 目标带、是否正确给出不确定/多解表达。
- 编码（乱码）自检。

任一红线触发时脚本以**退出码 1** 返回，可直接接 CI/流水线门禁。

**降采样（按目标配比）：** 在转换阶段用 `03_convert.py --rebalance` 触发：

```bash
python 03_convert.py --rebalance --image-root /your/abs/path/to/dataset
```

按目标配比（默认 existence 35% / location 25% / spatial 20% / attribute 15% / counting 5%）对各 `question_type` 桶降采样，**主要用于把 counting 压到目标比例**，避免高噪声计数题稀释主训练信号。这是 prompt 配比建议之外**真正可靠**的比例控制手段。

#### 本项目提供的人工审批工具

为支撑上述抽样人工质检，仓库内置了一个轻量本地审阅工具 `02_review.py`（逻辑在 `src/review.py`），是流水线的最后一环、也是数据进入 SFT 训练前**必须**走过的一步——自动质检只能过滤格式与硬规则错误，对幻觉、错答、计数误差等只能依赖人工判定。

快速启动：

```bash
python 02_review.py
# 浏览器打开 http://127.0.0.1:8008
```

完整使用说明（参数、界面功能、主键规则、下游消费）见 [RUN_GUIDE.md → 人工审批前端](RUN_GUIDE.md#人工审批前端)。

## 推荐输出目录

```text
sft_data/
├── raw/
│   ├── structured_desc.jsonl
│   ├── questions.jsonl
│   └── answers_raw.jsonl
├── processed/
│   ├── sft_vqa_distill_silver.jsonl
│   └── sft_vqa_distill_gold.jsonl
├── audit/
│   ├── audit_sample.jsonl
│   ├── audit_sheet.csv
│   └── human_reviewed.csv
├── rejected/
│   ├── format_fail.jsonl
│   ├── hallucination_risk.jsonl
│   └── unanswerable.jsonl
└── reports/
    ├── sft_stats.json
    └── audit_report.json
```

推荐分级：

- `gold`：人工审核通过或高置信规则通过的数据，可用于加权训练或 sanity check。
- `silver`：自动规则校验通过，且所在抽检桶质量达标的主训练数据。
- `rejected`：格式错误、明显幻觉、重复或无训练价值的数据。

## 采样策略

不要直接按文件顺序取前 N 张。推荐使用固定随机种子的分层采样。

分层维度：

- 数据集：EBD / LEVIR-CD+ / SECOND。
- 场景类型：城市、乡村、农田、水体、植被、工业区、灾后场景。
- 复杂度：简单 / 中等 / 复杂。
- 地物密度：低 / 中 / 高。

每次采样应记录：

- `seed`
- 原始候选池数量
- 各数据集采样数量
- 样本图像路径
- 采样时间

这样后续重跑和对比实验才能复现。

## Prompt 约束

描述、问题阶段仍保留较强约束；**答案阶段是纯蒸馏，只留最后一条护栏**（看不清就说不确定），不规定输出格式。各阶段共同保留的底线：

```text
只允许依据图像中可见内容回答。
不要推断图像外信息。
不要给出行政、工程、灾害治理建议。
不要编造精确坐标、面积、距离或数量。
如果目标不清晰，必须说明“不确定”或“无法确认”。
```

推荐答案风格（自然口吻、简洁、基于可见内容，**无格式标签**）：

```text
图像中央偏右那条浅灰色道路两侧分布着多个规则矩形屋顶，属于沿道路分布的建筑。
```

不推荐答案风格：

```text
该区域可能是城市扩张导致的居民区建设，建议加强规划管理并开展后续监测。
```

这类内容超出了 VQA 的图像证据范围，容易污染训练目标。

## 环境依赖

```bash
pip install openai tqdm
```

## vLLM 部署

教师模型需要以 OpenAI 兼容接口方式部署。默认服务地址为 `http://localhost:8001`。

实测可用的部署命令（Qwen3.5-27B，2 卡 TP）：

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

关键参数说明：

- `--mm-encoder-tp-mode data`：多模态视觉编码器走 data parallel，多卡下吞吐更高。
- `--mm-processor-cache-type shm`：图像预处理结果走共享内存缓存，重复访问同图时命中。
- `--reasoning-parser qwen3`：启用 Qwen3 系列的思考链解析；流水线侧可通过 `enable_thinking=False` 关闭思考以避免输出截断（详见 [常见问题排查](RUN_GUIDE.md#常见问题排查)）。
- `--enable-prefix-caching`：相同前缀的 prompt 复用 KV cache，对本项目这种 prompt 模板高度重复的场景显著提速。
- `--max-model-len 8192`：图像 token + prompt + 答案够用；如果遇到超长 prompt 报错，再增大。

或使用 `swift`：

```bash
swift deploy --model Qwen/Qwen3.5-27B --port 8001 --infer_backend vllm
```

## 运行方式

```bash
python construct_vqa_data.py --base-url http://localhost:8001 --concurrency 4 --retries 3
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 自动探测 | 指定模型名，也可通过 `$VLLM_MODEL` 环境变量设置 |
| `--base-url` | `http://localhost:8001` | vLLM 服务地址 |
| `--concurrency` | `4` | 并发请求数 |
| `--retries` | `3` | 单条记录失败后的最大重试次数 |

## 断点续传

构建脚本应将输出 JSONL 视为完成状态来源：

1. 启动时读取已有输出文件。
2. 提取每条记录的 `images[0]` 作为唯一标识。
3. 从待处理图像列表中过滤已完成样本。
4. 每生成成功一条就立即追加写入并 `flush`。
5. 失败样本不写入主输出，下次运行自动重试。

注意：断点续传依赖图像路径字符串。移动数据集目录会导致旧路径无法匹配，已完成样本可能被重新生成。

## 数据质量原则

纯蒸馏的 VQA SFT 数据应遵循以下原则：

- 问题短而明确。
- 答案只基于图像可见内容。
- 答案直接回答问题。
- 不把遥感报告当作 VQA 答案。
- 不把描述阶段的幻觉继续传递到问答阶段。
- 不为了数量牺牲问题质量。

本项目推荐的 SFT 冷启动目标是：模型能够先定位问题关注对象，再观察图像证据，最后给出简洁答案；后续再通过 GeoChat_Instruct 等数据进行 GRPO，提高遥感图像识别准确率。

---

## 快速开始

### 1. 准备数据集

将数据集放在 `dataset/` 目录下，支持：
- **EBD**: `EBD/{EVENT}/images/*_post_disaster.*`
- **LEVIR-CD+**: `LEVIR-CD+/{train,test}/B/*.png`
- **SECOND**: `SECOND/{train,test}/im2/*.png`

### 2. 启动vLLM服务

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.5-27B \
  --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size 2 --port 8001
```

### 3. 四步主线

```bash
# ① 生成基础数据（三阶段并行）+ 结束后自动体检报告
python 01_generate.py --num-images 100

# ② 人工审批（浏览器打开 http://127.0.0.1:8008，发布前必经）
python 02_review.py

# ③ 转 Unsloth 训练集 + 切 train/val（纯蒸馏下默认全保留；可选 --rebalance 降采样）
#    在 Linux 训练机上跑，--image-root 传图片真实根目录
python 03_convert.py --image-root /your/abs/path/to/dataset --check-images

# ④ Qwen3-VL 视觉 SFT（仅 Linux+GPU）。超参直接改 04_train.py 顶部各 section 的常量
python 04_train.py
```

`01_generate.py` 内部三阶段并行推进，支持**断点续传**（按图像绝对路径识别，中断后重跑自动跳过已处理内容）；结束后自动跑一次只读体检（`--no-health-check` 可关）。

**默认采样**：

| 数据集 | 采样数量 |
|--------|---------|
| LEVIR-CD+ | 500 |
| SECOND | 500 |
| **合计** | **1000 张图像** |

每张图像生成 **2-5 个 VQA 问题**，由模型根据图像内容自适应：内容简单的图只生成 2 个，内容丰富的图最多 5 个；解析后超过 5 个会硬截断，避免凑数。预计最终生成 **2000–5000 个 SFT 样本**。

> `01_generate.py` 不传 `--num-images` 时走上面的默认采样。默认不启用 EBD（灾后场景）；如需纳入 EBD，参考 `RUN_GUIDE.md` 中的"完整运行流程"章节。

详细参数说明请查看 `RUN_GUIDE.md`。

---

## 🧪 小规模冒烟

批量运行前，先用极小样本量把整条链路走通：

```bash
# 生成阶段：每个数据集各抽 2 张，验证 VLM 连接 / 三阶段 / 自动质检 / 体检报告
python 01_generate.py --samples-per-dataset 2

# 转换阶段：不依赖 GPU，验证格式转换 / 质检过滤 / 路径重映射
python 03_convert.py --image-root /your/abs/path/to/dataset

# 训练阶段：把 04_train.py 里的 `max_steps = 30` 那行取消注释，先跑几十步冒烟（仅 Linux+GPU）
python 04_train.py
```

走通后再放大 `--num-images`（生成）/ 把 `max_steps` 注释回去用 `num_train_epochs`（训练）正式跑。
