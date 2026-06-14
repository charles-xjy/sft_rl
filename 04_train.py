#!/usr/bin/env python3
# ============================================================================
# Qwen3-VL 视觉 SFT 训练（扁平版，照搬 Unsloth 官方 Qwen3-VL (8B) Vision notebook 结构）
#
#   仅 Linux + NVIDIA GPU 运行。读 03_convert.py 产出的 outputs/sft/train.jsonl。
#   按「① 加载模型 → ② 挂 LoRA → ③ 准备数据 → ④ 训练器 → ⑤ 训练 → ⑥ 保存」六步走，
#   每个参数都在所属步骤里就近注释，方便对照 notebook 学习。
#
#   要改超参，直接改下面各 section 里的值即可（没有命令行参数，刻意保持扁平）。
#   先把 max_steps 那行取消注释跑 30 步冒烟，确认能通了再注释回去正式训练。
# ============================================================================

from PIL import Image
from datasets import load_dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from swanlab.integration.transformers import SwanLabCallback   # ← SwanLab 可视化

# 基座与路径（按需修改）
MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"  # 官方 4bit 量化版，显存友好
TRAIN_FILE = "outputs/sft/train.jsonl"
OUTPUT_DIR = "outputs/qwen3vl-sft-lora"

# SwanLab 看板的项目 / 实验名（按需修改）
SWANLAB_PROJECT = "qwen3vl-rs-sft"
SWANLAB_EXPERIMENT = "qwen3vl-8b-lora"


# ============================================================================
# ① 加载模型  ★ 省显存/省时间几乎全发生在这一步 ★
# ============================================================================
# 下面的训练循环用的是 trl 的 SFTTrainer（标准接口，看不出 unsloth）。unsloth 不替换
# 训练循环，它替换的是「被训练的 model 本身」：FastVisionModel.from_pretrained 返回的不是
# 普通 HF 模型，而是在加载时做了「手术」的版本——
#   1) 4bit 量化（配合 ...-bnb-4bit 仓库 + load_in_4bit）：权重从 ~16GB 压到 ~5GB，8B 才能单卡训；
#   2) 把注意力 / MLP / LayerNorm / RoPE / 交叉熵换成 unsloth 手写的 Triton 融合内核：
#      更少的显存读写 → 更快；融合交叉熵不实体化完整 logits 大张量（VLM vocab 很大，省得多）；
#   3) use_gradient_checkpointing="unsloth"：比 HF 默认的梯度检查点更省激活显存（约 -30%）。
# 之后这个 model 交给 trl 照常训——同一份 SFTTrainer 代码，喂普通模型就是普通速度/显存，
# 喂这个 model 就快、就省。加载时会打印 "Unsloth ...: Fast Qwen3-VL patching" banner 作证。
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit = True,                     # 4bit 量化，8B 单卡可训；显存充裕可设 False 走 16bit
    use_gradient_checkpointing = "unsloth",  # Unsloth 省显存的梯度检查点实现，几乎必开
)


# ============================================================================
# ② 挂 LoRA（冻结基座，只训练少量适配器参数）
# ============================================================================
# 这是第二处省显存的来源：只训练 LoRA 适配器，优化器(Adam)的动量/方差只为这一小撮参数存，
# 而不是给整个 8B 存——优化器状态显存因此降一个数量级。
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,   # 训练视觉塔——遥感图与自然图差异大，建议保持开
    finetune_language_layers   = True,   # 训练语言层
    finetune_attention_modules = True,   # 训练注意力模块
    finetune_mlp_modules       = True,   # 训练 MLP 模块
    r = 16,             # LoRA 秩 = 可训练容量；几千条数据 16 够用，欠拟合再升 32
    lora_alpha = 16,    # LoRA 缩放，惯例设成 = r
    lora_dropout = 0,   # Unsloth 对 0 有优化路径，最快；想抗过拟合可设 0.05
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)


# ============================================================================
# ③ 准备数据：转成 Unsloth 视觉对话格式
# ============================================================================
# 我们的 train.jsonl 每行：{"image": 图片路径, "question": 问题, "answer": "<analysis>..<answer>.."}
# Unsloth 要的是 content 列表 + 真正的 PIL 图对象，所以这里把路径 Image.open 成图。
# （路径已由 03_convert.py 重映射到训练机真实目录，直接打开即可。）
dataset = load_dataset("json", data_files = TRAIN_FILE, split = "train")

instruction_is_per_sample = True  # 我们的指令就是每条的 question，不是固定一句

def convert_to_conversation(sample):
    return {
        "messages": [
            { "role": "user",
              "content": [
                {"type": "text",  "text":  sample["question"]},
                {"type": "image", "image": Image.open(sample["image"]).convert("RGB")},
              ]},
            { "role": "assistant",
              "content": [
                {"type": "text", "text": sample["answer"]},
              ]},
        ]
    }

# 注意：这里是 eager 列表，会把所有图片读进内存。几千张没问题；
# 数据规模很大（几十万）时改成 dataset.with_transform(...) 惰性读图更省内存。
converted_dataset = [convert_to_conversation(sample) for sample in dataset]


# ============================================================================
# ④ 训练器
# ============================================================================
FastVisionModel.for_training(model)   # 切到训练模式

# SwanLab 可视化：把 loss / 学习率 / 显存等曲线实时记录到 swanlab 看板。
# 首次使用先在终端跑 `swanlab login` 贴上 API key（或设环境变量 SWANLAB_API_KEY）；
# 想完全离线、只在本机看，把下面加一行 mode="local" 即可。
# config 里的超参只是为了在看板上展示，方便不同实验对比，不影响训练。
swanlab_callback = SwanLabCallback(
    project = SWANLAB_PROJECT,
    experiment_name = SWANLAB_EXPERIMENT,
    config = {
        "model": MODEL_NAME,
        "lora_r": 16,
        "epochs": 2,
        "effective_batch": 2 * 8,   # per_device_train_batch_size * gradient_accumulation_steps
        "learning_rate": 2e-4,
    },
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),  # 视觉专用，自动拼图像 token
    train_dataset = converted_dataset,
    callbacks = [swanlab_callback],   # ← 接入 SwanLab；report_to 保持 "none" 由 callback 负责
    args = SFTConfig(
        # ---- 批大小：有效 batch = 下面两者相乘 = 16 ----
        per_device_train_batch_size = 2,    # 单卡一次几条；OOM 就降到 1
        gradient_accumulation_steps = 8,    # 累积几步再更新；降了上面就升这个，保持乘积 16

        # ---- 训练时长 ----
        num_train_epochs = 2,               # ~3700 条偏小，从 2 起，看 loss 再加到 3
        # max_steps = 30,                   # 取消注释 → 覆盖 epochs，只跑 30 步快速冒烟
        warmup_steps = 5,

        # ---- 学习率 ----
        learning_rate = 2e-4,               # LoRA 标准值；全参微调才用 1e-5 量级，LoRA 用错几乎不学
        lr_scheduler_type = "linear",
        weight_decay = 0.001,

        # ---- 精度 / 优化器 ----
        optim = "adamw_8bit",               # 8bit 优化器，省显存
        bf16 = True,                        # Ampere(30/40系/A100)以上支持；老卡(V100)改成 fp16 = True

        # ---- 日志 / 保存 ----
        logging_steps = 5,
        save_steps = 100,
        output_dir = OUTPUT_DIR,
        seed = 3407,
        report_to = "none",

        # ---- 视觉 SFT 必需：别让 TRL 按纯文本流程处理数据集，否则会丢图像列 ----
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,                  # 图像 token + 问题 + 答案的总长度上限
    ),
)


# ============================================================================
# ⑤ 训练
# ============================================================================
trainer.train()


# ============================================================================
# ⑥ 保存 LoRA 适配器
# ============================================================================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[saved] LoRA 适配器 -> {OUTPUT_DIR}")

# 需要合并成完整 16bit 权重（方便 vLLM 直接部署）时，取消下面这行注释：
# model.save_pretrained_merged(OUTPUT_DIR + "_merged", tokenizer, save_method = "merged_16bit")
