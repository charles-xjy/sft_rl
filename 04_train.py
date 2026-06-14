#!/usr/bin/env python3
# ============================================================================
# Qwen3-VL 视觉 SFT 训练（扁平版，照搬 Unsloth 官方 Qwen3-VL Vision notebook 结构）
#
#   仅 Linux + NVIDIA GPU 运行。读 03_convert.py 产出的 train.jsonl / val.jsonl。
#   流程：⓪ 选卡 → ① 加载模型 → ② 挂 LoRA → ③ 准备数据(含验证集) →
#         ④ 训练器 → ⑤ 训练 → ⑥ 保存。每个参数都在所属步骤就近详注。
#
#   改超参直接改下面各 section 的值（刻意保持扁平，无命令行参数）。
#   先把 max_steps 那行取消注释跑几十步冒烟，确认能通了再注释回去正式训练。
# ============================================================================

# ============================================================================
# ⓪ 选卡：必须在 import torch / unsloth 之前设置，否则不生效
# ============================================================================
import os
# 只让本进程看到第 1 号物理 GPU（你的 GPU 0 在跑桌面且偏热，1/2 空闲）。
# 设了之后进程内这张卡会被重新编号为 cuda:0，属正常现象。
# 想用别的卡改这里；想用多卡（本脚本不支持）需另走 accelerate/DDP。
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import Image
from datasets import load_dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from swanlab.integration.transformers import SwanLabCallback   # SwanLab 可视化

# ---- 基座与路径（按需修改）----
# ModelScope 下载到本地的原版 Qwen3-VL-2B-Instruct（非量化）；用绝对路径，避免联网拉 HF。
MODEL_NAME = "/home/charles/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct"
TRAIN_FILE = "outputs/sft/train.jsonl"   # 训练集（03_convert.py 产出）
VAL_FILE   = "outputs/sft/val.jsonl"     # 验证集（03_convert.py 用 --val-ratio 切出来的）
OUTPUT_DIR = "outputs/qwen3vl-sft-lora"  # LoRA 适配器 / checkpoint 输出目录

# ---- SwanLab 看板的项目 / 实验名（按需修改）----
SWANLAB_PROJECT = "qwen3vl-rs-sft"
SWANLAB_EXPERIMENT = "qwen3vl-2b-lora"


# ============================================================================
# ① 加载模型  ★ 省显存/省时间几乎全发生在这一步 ★
# ============================================================================
# trl 的 SFTTrainer 只是标准训练循环，看不出 unsloth。unsloth 不替换训练循环，它替换的是
# 「被训练的 model 本身」：from_pretrained 返回的不是普通 HF 模型，而是加载时被「手术」过的：
#   1) 注意力/MLP/LayerNorm/RoPE/交叉熵 换成 unsloth 手写 Triton 融合内核 → 更快、显存读写更少；
#      融合交叉熵不实体化完整 logits 大张量（VLM 词表大，这项省得多）；
#   2) use_gradient_checkpointing="unsloth" → 比 HF 默认梯度检查点更省激活显存（约 -30%）。
# 不量化（下面 4bit/8bit 都 False）时就靠这两项省；开 4bit 会再多「权重量化」一层=QLoRA。
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit = False,                    # 4bit 量化权重？False=普通 LoRA(16bit 基座)；True=QLoRA(省显存,精度略损)
    load_in_8bit = False,                    # 8bit 量化权重？同上，关
    use_gradient_checkpointing = "unsloth",  # 梯度检查点：用重算换显存。"unsloth"=优化版，几乎必开
    # max_seq_length = 2048,                 # 可显式设上下文上限；不传则自动探测，一般不用管
)


# ============================================================================
# ② 挂 LoRA（冻结基座权重，只训练旁挂的小适配器矩阵）
# ============================================================================
# LoRA：原权重 W 冻住，在旁边加一对小矩阵 A×B 来学增量，训练量从「整个模型」降到「几个小矩阵」。
# 第二处省显存就在这：优化器(Adam)的动量/方差只为这一小撮适配器参数存，不为整个基座存。
model = FastVisionModel.get_peft_model(
    model,
    # ↓ 选「模型的哪些部分」挂 LoRA。VLM 由视觉塔+语言模型两半组成，可分别开关。
    finetune_vision_layers     = True,   # 训视觉塔——遥感图与自然图差异大，建议开；OOM 时第一个考虑关
    finetune_language_layers   = True,   # 训语言层（负责生成 analysis/answer），一般开
    finetune_attention_modules = True,   # LoRA 挂到注意力投影(q/k/v/o)，开
    finetune_mlp_modules       = True,   # LoRA 挂到 MLP(gate/up/down)，开
    # ↓ LoRA 本身的超参
    r = 16,             # LoRA 秩 = 可训练容量。越大越能学但越占显存/越易过拟合；几千条数据 16 够，欠拟合升 32
    lora_alpha = 16,    # 缩放系数，等效放大≈alpha/r。惯例设成 = r（这里 16/16=1 倍）
    lora_dropout = 0,   # LoRA 层 dropout。0 最快(unsloth 有优化路径)；想抗过拟合设 0.05
    bias = "none",      # 是否一起训 bias。"none" 最省显存，标配
    random_state = 3407,    # 随机种子（LoRA 初始化等），固定可复现
    use_rslora = False,     # rank-stabilized LoRA（大 r 时缩放更稳）。小 r 用不上
    loftq_config = None,    # LoftQ 量化初始化，仅量化场景用；不量化就 None
)


# ============================================================================
# ③ 准备数据：转成 Unsloth 视觉对话格式（训练集 + 验证集都转）
# ============================================================================
# train.jsonl / val.jsonl 每行：{"image": 路径, "question": 问题, "answer": "<analysis>..<answer>.."}。
# Unsloth 要的是 content 列表 + 真正的 PIL 图对象，所以这里把图片路径 Image.open 成图。
# 路径已由 03_convert.py 重映射到训练机真实目录，直接打开即可。
def convert_to_conversation(sample):
    """把一行样本转成 Unsloth 视觉对话：user=[问题文本, 图]，assistant=[答案]。"""
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

# 训练集
train_dataset = load_dataset("json", data_files = TRAIN_FILE, split = "train")
# 验证集：03_convert.py 默认按 --val-ratio 0.02 切了一份 val.jsonl。
# 它不参与训练，只在训练途中定期算 loss，用来看是否过拟合（train loss 降但 val loss 升=过拟合）。
val_dataset = load_dataset("json", data_files = VAL_FILE, split = "train")

# 注意：这里是 eager 列表，会把图片读进内存。几千张没问题；
# 数据规模很大（几十万）时改成 dataset.with_transform(...) 惰性读图更省内存。
converted_train = [convert_to_conversation(s) for s in train_dataset]
converted_val   = [convert_to_conversation(s) for s in val_dataset]


# ============================================================================
# ④ 训练器
# ============================================================================
FastVisionModel.for_training(model)   # 把模型切到训练模式（开启梯度等）

# ---- SwanLab 可视化 ----
# 把 loss / 学习率 / 显存 / 验证 loss 等曲线实时记录到 swanlab 看板。
# 首次使用先在终端 `swanlab login` 贴 API key（或设环境变量 SWANLAB_API_KEY）；
# 想完全离线只在本机看，给下面加一个 mode="local"。
# config 里的超参仅用于看板展示、方便不同实验对比，不影响训练。
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
    data_collator = UnslothVisionDataCollator(model, tokenizer),  # 视觉专用 collator，自动拼图像 token
    train_dataset = converted_train,   # 训练集
    eval_dataset  = converted_val,     # 验证集（接上后才有 val loss 曲线）
    callbacks = [swanlab_callback],    # 接入 SwanLab；report_to 保持 "none"，避免重复 run
    args = SFTConfig(
        # ---------- 批大小：有效 batch = 下面两者相乘 = 16 ----------
        per_device_train_batch_size = 2,    # 单卡一次喂几条。直接决定显存峰值，OOM 先降到 1
        gradient_accumulation_steps = 8,    # 累积几个 batch 再更新一次。降了上面就升这个，保持乘积 16
        per_device_eval_batch_size  = 2,    # 验证时一次喂几条（不反传，可比训练大些）

        # ---------- 训练时长 ----------
        num_train_epochs = 1,               # 过几遍数据。~3700 条偏小，从 2 起，看 val loss 再加到 3
        # max_steps = 30,                   # 取消注释 → 覆盖 epochs，只跑 30 步快速冒烟
        warmup_steps = 20,                   # 开头几步学习率从 0 线性升到设定值，避免一上来就乱跳

        # ---------- 学习率 ----------
        learning_rate = 2e-4,               # LoRA 标准值；全参微调才用 1e-5 量级，LoRA 给小了几乎不学
        lr_scheduler_type = "linear",       # 学习率衰减曲线：线性降到 0（也常用 "cosine"）
        weight_decay = 0.001,               # 权重衰减(L2 正则)，抑制过拟合，小值即可

        # ---------- 精度 / 优化器 ----------
        optim = "adamw_8bit",               # 8bit 版 AdamW：优化器状态用 8bit 存，省显存，几乎无损
        bf16 = True,                        # 用 bfloat16 计算。Ada/30/40系/A100 支持；老卡 V100 改 fp16=True

        # ---------- 验证：定期在 val 集上算 loss ----------
        eval_strategy = "steps",            # 按步数触发验证（另有 "epoch" / "no"）
        eval_steps = 10,                    # 每 50 步在验证集上评一次，画出 val loss 曲线

        # ---------- 日志 / 保存 ----------
        logging_steps = 5,                  # 每 5 步打一次 train loss（也推给 SwanLab）
        save_steps = 100,                   # 每 100 步存一次 checkpoint 到 output_dir
        save_total_limit = 3,               # 最多保留 3 个 checkpoint，旧的自动删，省磁盘
        output_dir = OUTPUT_DIR,            # checkpoint / 中间产物输出目录
        seed = 3407,                        # 全局随机种子，复现用
        report_to = "none",                 # 日志上报方式：none，由上面的 SwanLabCallback 负责

        # ---------- 视觉 SFT 必需（不是调参，是别让 TRL 把多模态当纯文本处理）----------
        remove_unused_columns = False,                      # 否则 TRL 会把 messages/图像列当没用删掉
        dataset_text_field = "",                            # 告诉 TRL 别去找纯文本字段
        dataset_kwargs = {"skip_prepare_dataset": True},    # 跳过 TRL 文本预处理，交给视觉 collator
        max_length = 2048,                                  # 单条序列总长上限(图像token+问题+答案)，超了截断
    ),
)


# ============================================================================
# ⑤ 训练
# ============================================================================
trainer.train()


# ============================================================================
# ⑥ 保存 LoRA 适配器
# ============================================================================
# 只存 LoRA 适配器（很小，几十 MB），不含基座；推理时基座 + 适配器一起加载。
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[saved] LoRA 适配器 -> {OUTPUT_DIR}")

# 需要合并成完整 16bit 权重（方便 vLLM 直接部署）时，取消下面这行注释：
# model.save_pretrained_merged(OUTPUT_DIR + "_merged", tokenizer, save_method = "merged_16bit")
