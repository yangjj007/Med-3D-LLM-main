# 3D-VL 项目进度与上手指南

本文档面向**新接手或协作的开发者**，汇总当前对话与实现结论、项目现状、可执行路径，以及**已知卡点与注意事项**，便于快速上手和避坑。

---

## 一、对话与需求总结

### 1.1 目标

- **3D-VL 对齐**：把 3D 形状（SDF → VAE 编码）与视觉语言模型（Qwen2-VL / Qwen3-VL）对齐，实现「给定 3D → 生成描述」或「给定 3D → 重建 3D」等任务。
- **两条技术路线**：
  - **连续特征 + Projector**：VAE 输出连续向量，经 Projector 映射到 VL 隐空间，与文本拼接输入。
  - **离散 3D Token**：VAE 输出码本索引，扩词表为 `<mesh_0>`…`<mesh_8191>`，无 Projector，直接拼成 token 序列输入 VL。

### 1.2 关键决策：从「固定 512」到「变长 3D」

- 早期实现：VAE 编码后做 **8×8×8 空间池化**，得到**固定 512 个** mesh token 再进 LLM。
- 用户澄清：**8³ 是接入 LLM 的代码里做的池化分辨率，不是 VAE 原生输出**；希望「VAE 出来多少点就用多少点」，不再做降采样，直接进 LLM。
- 因此增加 **Variable-Length 3D Tokenization（变长 3D）**：
  - 保留 VAE 输出的全部点（约 8k～12k），不做 8³ 池化。
  - **莫顿码排序**：按 (x,y,z) Z-Order 排序，序列相邻 token 在 3D 空间相邻。
  - **动态 Padding + attention_mask**：batch 内按最长序列 pad，不设全局截断。
  - **软性上限**：仅当点数 > 15000 时用 FPS 降到 15000，避免异常长序列 OOM。
  - **显存策略**：Flash Attention 2、梯度检查点、DeepSpeed ZeRO-2（可选）。

### 1.3 VAE 坐标范围澄清

- **VAE 输入**：SDF 网格 **512³**（`sparse_index` 范围 0..511）。
- **当前 trellis Encoder**：3 次 stride-2 下采样 → 输出 **64³ latent**，坐标 **0..63**（不是 8³，也不是 512³）。
- **8³**：是「接入 LLM」时在 `spatial_pool_3d` 里做的**再池化**（64³→8³→512 token），属于固定长度方案。
- **配置**：Morton 与变长逻辑使用 **`coord_max_3d=64`** 对应当前 VAE；若将来改用「输出 512³ 坐标」的 VAE，再改为 512。

---

## 二、项目当前状态

### 2.1 代码结构（与 3D-VL 相关）

| 路径 | 作用 |
|------|------|
| `vae_qwen3vl/train_finetune.py` | 训练入口；支持离散/变长、LoRA、warmup/sft、DeepSpeed |
| `vae_qwen3vl/dataset_sdf_caption.py` | SDF+Caption 数据集；`collate_sdf_caption_discrete` 支持固定 8³ 与变长 |
| `vae_qwen3vl/spatial_pool_3d.py` | 8×8×8 池化 → 固定 512 mesh token |
| `vae_qwen3vl/variable_length_3d.py` | 变长：Morton 排序、FPS 软截断、变长序列生成 |
| `vae_qwen3vl/tokenizer_3d.py` | mesh token 扩表、embedding resize 与初始化 |
| `vae_qwen3vl/model.py` | Qwen3VLWith3DBranch；离散时无 Projector，走 input_ids/labels |
| `vae_qwen3vl/eval_3d_vl.py` | 评估；可识别离散 run（tokenizer_final）走离散生成 |
| `scripts/run_3d_align_train.py` | 从 YAML 写 accelerate 配置并启动训练；支持 `use_deepspeed` |

### 2.2 配置与入口

| 文件 | 用途 |
|------|------|
| `configs/3d_align_train.yaml` | 通用训练配置（连续/离散、多卡）；`use_deepspeed` 可选 |
| `configs/3d_align_train_discrete_warmup.yaml` | 离散 Stage1 专用（warmup、单卡） |
| `configs/3d_align_train_variable_length.yaml` | **变长 3D** 专用：变长 + Flash Attn + 梯度检查点 + DeepSpeed 默认开 |
| `configs/accelerate.yaml` | 由 `run_3d_align_train.py` 根据 YAML 动态生成 |

### 2.3 已实现功能清单

- [x] 连续特征路径：VAE → Projector → VL，LoRA/全参数可配  
- [x] 离散固定 512：8³ 池化 + mesh 扩表，两阶段（warmup → sft）  
- [x] **变长 3D**：Morton 排序、FPS 软上限 15000、动态 padding、attention_mask  
- [x] Flash Attention 2、梯度检查点、梯度累积  
- [x] DeepSpeed ZeRO-2 可选（`use_deepspeed: true`），降低每卡显存  
- [x] 离散评估：检测 `tokenizer_final`，resize 词表，离散生成  
- [x] `coord_max_3d` 统一为 64（当前 VAE 64³ latent）

---

## 三、快速上手：我该跑哪条路？

### 3.1 环境与前置

- **Python**：3.8+，`torch`、`transformers`、`accelerate`、`peft`、`pyyaml`（见项目 requirements）。
- **必设**：`export SPARSE_BACKEND=torchsparse`（训练/评估前都要）。
- **必备**：
  - **VAE**：`vae_config` + `vae_ckpt`（离散/变长必填）。
  - **数据**：`data_dir` 下 `*.npz`（SDF）+ `metadata.csv`（含 `captions`）。
  - **VL 模型**：`vl_model` 指向 Qwen2-VL / Qwen3-VL 路径或 HF 名。

### 3.2 三条典型命令

**1）离散固定 512，Stage1 warmup（单卡）**

```bash
export SPARSE_BACKEND=torchsparse
python scripts/run_3d_align_train.py --config configs/3d_align_train_discrete_warmup.yaml 2>&1 | tee align_debug.log
```

**2）变长 3D（VAE 出多少点就多少点，不池化）**

```bash
export SPARSE_BACKEND=torchsparse
python scripts/run_3d_align_train.py --config configs/3d_align_train_variable_length.yaml 2>&1 | tee align_debug.log
```

**3）多卡并行定位（TP/SP/通信详细日志）**

推荐在排查 NCCL 超时、序列并行或张量并行问题时使用：

```bash
export SPARSE_BACKEND=torchsparse
export PARALLEL_DEBUG=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
python scripts/run_3d_align_train.py --config configs/3d_align_train_variable_length.yaml 2>&1 | tee align_debug.log
```

若需要进一步看每层 patch 与梯度同步细节（日志非常大）：

```bash
export PARALLEL_DEBUG_VERBOSE=1
python scripts/run_3d_align_train.py --config configs/3d_align_train_variable_length.yaml 2>&1 | tee align_debug_verbose.log
```

### 3.3 配置要点速查

| 目的 | 配置 |
|------|------|
| 用变长、不池化 | `use_discrete_3d_tokens: true` + `use_variable_length_3d_tokens: true` |
| 降显存（变长） | `use_flash_attn_2: true`、`use_gradient_checkpointing: true`、`use_deepspeed: true`、`batch_size: 1`、`gradient_accumulation_steps: 8` |
| Morton 坐标范围 | 当前 VAE：`coord_max_3d: 64`；若 VAE 输出 512³ 则 512 |
| 软截断长度 | `max_safe_3d_length: 15000`（仅超过此点数才 FPS） |

---

## 四、已知卡点与注意事项

### 4.1 环境与依赖

- **SPARSE_BACKEND=torchsparse**：未设置时稀疏运算可能走错后端或报错，**所有训练/评估命令前都要 export**。
- **Flash Attention 2**：`use_flash_attn_2: true` 需安装 `flash-attn`（部分环境需从源码编译）。
- **DeepSpeed**：`use_deepspeed: true` 需 `pip install deepspeed`；多卡时 ZeRO-2 可显著降低每卡显存。

### 4.2 数据与 VAE

- 离散/变长路径**必须**提供 `vae_config` 和 `vae_ckpt`，否则 collate 会报错。
- `data_dir` 下 npz 的 `sparse_index` 需与配置中的分辨率一致（如 512³）；metadata 需有 `captions` 列。

### 4.3 显存与性能

- **变长序列**约 8k～12k token，显存占用大：建议 `batch_size=1` + `gradient_accumulation_steps` 4～8，并开启 Flash Attn、梯度检查点或 DeepSpeed。
- **梯度检查点**：省显存但会变慢；若显存够可关掉（`use_gradient_checkpointing: false`）换速度。
- **FPS 软截断**：当某样本点数 > 15000 时在 CPU 上做 FPS，点数极大时可能较慢；可适当调低 `max_safe_3d_length` 或先过滤异常样本。

### 4.4 训练与评估

- 离散 run 的产出是 **tokenizer（含 mesh 词条）** + 可选 **LoRA**，**没有** `projector_final.pt`；评估脚本通过是否存在 `tokenizer_final` 判断离散 run。
- 两阶段：先 **warmup**（只训 embed + lm_head），再 **sft**（LoRA + embed）；Stage2 可从 Stage1 的 tokenizer 继续，LoRA 按需加载。

### 4.5 loss=NaN/Inf 排查与解决

- **现象**：日志出现 `[MEM WARN] step=N loss=NaN/Inf (nan)！检查 lr/数据。`
- **常见原因**：
  1. **labels 全为 -100**：变长 3D 下序列很长，若 `max_length_variable` 或 tokenizer 的 `max_length` 不足，会把 **assistant 回复整段截掉**，导致该 batch 没有监督 token，交叉熵分母为 0 → NaN。
  2. **学习率过大**：BF16 + 大 lr 易导致数值不稳定，可先降到 `2e-5` 验证。
- **已加调试与防护**：
  - 训练前 5 步会打印 `[LOSSCHK] step=N valid_label_tokens=X/Y lr=...`；若某步 `valid_label_tokens=0` 会**跳过该 batch** 并打印提示。
  - 出现 NaN/Inf 时会打印 `valid_label_tokens`、`seq`、`lr`，并**跳过该步**（zero_grad + continue），避免污染后续训练。
  - collate 中若某样本 `valid_label_tokens=0` 会打印 `[LOSSCHK WARN] ... per_sample_valid=...`，便于定位是哪个样本被截断。
- **建议操作**：
  1. 在配置中增大 `max_length_variable`（如 32768 或更大），确保「user 3D + prompt + assistant 回复」不被截断。
  2. 开启 `debug_loss: true`（或看前几步的 `[LOSSCHK]`）确认 `valid_label_tokens` 是否大于 0。
  3. 若仍 NaN，可暂时将 `lr` 降到 `2e-5` 或 `1e-5` 验证是否稳定。

### 4.6 报错与日志

- 训练报错会写入 `configs/.train_error_rank*.txt`；完整日志在 `logs/` 下按时间戳的 `train_*_rank0.log` 等。
- 若在 collate 的 FPS 或 Morton 处卡住，多为单样本点数极大或 `coord_max_3d` 与 VAE 输出不一致，可先单卡、小 `max_samples` 复现。

---

## 五、相关文档索引

| 文档 | 内容 |
|------|------|
| `docs/3D-VL离散Token训练计划.md` | 离散/变长可执行命令、两阶段训练、评估命令、配置表 |
| `docs/3D-VL对齐策略与可行性评估.md` | 对齐策略、模型结构、连续路径说明 |
| `configs/3d_align_train.yaml` | 配置项注释（含变长、DeepSpeed、coord_max_3d） |

---

## 六、后续可做（可选）

- **ZeRO-3 / CPU offload**：显存仍不足时可在 DeepSpeed 配置中开 Stage-3 或 offload。
- **从 Stage1 继续 Stage2**：当前 Stage2 可从基座重新加载再挂 LoRA；若需从 Stage1 ckpt 连续训，需在脚本中加「加载上一阶段 ckpt」逻辑。
- **VAE 输出 512³**：若更换为「无下采样、输出 512³ 坐标」的 VAE，只需将 `coord_max_3d` 改为 512，并确认数据与 FPS/Morton 的数值范围一致。

---

*文档基于当前代码与对话整理，若配置或脚本有变更，以仓库内实际代码与最新文档为准。*
