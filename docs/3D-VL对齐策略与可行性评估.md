# 3D-VL 对齐策略与模型结构清单 · 可行性评估

本文档罗列当前 3D→文本对齐的**所有策略与模型结构设计**，并评估**是否能正常进行对齐**。

---

## 一、模型结构设计

### 1.1 整体流水线

```
3D 稀疏 SDF (batch)
    → VAE.Encode (冻结) → 离散 indices
    → codebook(indices) → (feats [N,16], coords [N,4])
    → prepare_3d_sequence (排序 / 截断或随机采样 / 填充) → (feats_seq, mask_3d, coords_seq)
    → Projector3D (可训) → embeds_3d [B, L, hidden_size]
    → 与 text_embeds 在序列维拼接 → combined_embeds [B, seq_3d + seq_text, hidden_size]
    → VL (Qwen2-VL / Qwen3-VL，可选 LoRA 可训) → causal LM loss
```

### 1.2 各模块设计

| 模块 | 设计要点 | 可训练性 |
|------|----------|----------|
| **VAE** | SparseSDFVQVAE，输出码本向量 16 维 + 格点坐标；推理时 `vae.eval()`，`requires_grad=False` | 冻结 |
| **3D 序列化** | 按 (batch_idx, z, y, x) 排序；`max_3d_tokens` 截断或 `truncate_mode="random_sample"` 随机采样；不足则 pad；输出固定长度 L 的 feats/mask/coords | 无参数 |
| **Projector3D** | latent_dim(16) → hidden_size(与 VL 一致)；1 层 Linear 或多层 MLP(GELU)；可选 3D 位置编码(sinusoidal/learned)；末尾 LayerNorm | 可训 |
| **VL 主体** | Qwen2-VL / Qwen3-VL，`inputs_embeds=combined_embeds`，不做图像分支 | 可选 LoRA 可训 |
| **LoRA** | target_modules: q_proj, v_proj, k_proj, o_proj；r, alpha, dropout 可配 | 可训 |

### 1.3 3D 与文本的对接方式

- **空间**：3D token 与文本 token 共用同一 `hidden_size`，在**序列维度拼接**，即 `[embeds_3d; text_embeds]`。
- **位置**：3D 作为**前缀**，先于 user prompt 与 assistant 回复；VL 的 causal attention 下，文本可 attend 到所有 3D token。
- **Labels**：3D 段与 user prompt 段在 `combined_labels` 中置为 `-100`，**仅对 assistant 回复部分（caption）计算交叉熵**，即监督「给定 3D+prompt，生成 caption」。

---

## 二、对齐策略清单

### 2.1 训练目标与监督信号

| 策略 | 实现方式 | 作用 |
|------|----------|------|
| **Causal LM 仅监督 caption** | `combined_labels` 前 `seq_3d + prompt_len` 为 -100，后面为 caption 的 token id | 让 VL 学会「看到 3D 前缀 + 用户问句后，生成描述」，不监督 3D 或 prompt 的编码 |
| **3D 作为 prefix embedding** | `combined_embeds = cat([embeds_3d, text_embeds], dim=1)`，一次性输入 VL | 与 LLaVA/Shikra 等「多模态前缀」做法一致，VL 通过 self-attention 利用 3D 信息 |

### 2.2 数据与 prompt 策略

| 策略 | 实现方式 | 作用 |
|------|----------|------|
| **SDF + 文本描述对** | SDF3DCaptionDataset：每个样本 (sparse_sdf, sparse_index) + caption(s)；collate 时合并 batch、构造 inputs_3d | 提供 3D–文本配对，caption 可多句随机选一 |
| **固定 user prompt** | 配置中 `prompt: "Describe this 3D shape in one sentence:"`，与 caption 组成 [user, assistant] 对话 | 统一任务形式，便于 chat 模板与生成时 add_generation_prompt |
| **Chat 模板** | tokenizer.apply_chat_template([user, assistant], add_generation_prompt=False)，再 tokenize/pad/truncate | 与 Qwen 对话格式一致，labels 中仅 assistant 部分非 -100 |

### 2.3 可训练参数与正则

| 策略 | 实现方式 | 作用 |
|------|----------|------|
| **只训 Projector** | 默认：仅 `model.projector` 的 `requires_grad=True`，VAE 与 VL 主体冻结 | 参数量小、稳定，Projector 学习「3D 潜空间 → VL 语义空间」的映射 |
| **可选 LoRA 微调 VL** | use_lora 时对 q_proj/v_proj/k_proj/o_proj 加 LoRA，仅 LoRA 参数可训 | 让 VL 适度适配「3D 前缀」的分布，提升对齐效果 |
| **梯度裁剪** | grad_clip=1.0，对可训参数做 clip_grad_norm_ | 防止梯度爆炸 |
| **Weight decay** | AdamW weight_decay=0.01 | 减轻过拟合 |
| **学习率与调度** | lr=5e-5，可选 cosine + warmup (warmup_ratio=0.05) | 稳定收敛 |

### 2.4 3D 表示与序列化策略

| 策略 | 实现方式 | 作用 |
|------|----------|------|
| **使用码本向量而非 encoder 连续输出** | feats = vae.vq.embeddings(indices) | 离散化与 VL 的 token 表示更一致，且与 Decode 重建一致 |
| **排序 (z,y,x)** | _sort_coords_feats 按 batch_idx, z, y, x 排序 | 序列顺序确定、空间连续，便于学习局部结构 |
| **截断方式** | truncate_mode: head（取前 L）或 random_sample（随机采 L） | 超长时控制显存；random_sample 保留空间覆盖、利于理解 |
| **3D 位置编码** | Projector 内可选 PositionEncoder3D(sinusoidal/learned)，out = proj(feats) + pos_encoder(coords) | 显式注入格点坐标，加强空间关系 |
| **Projector 出口 LayerNorm** | Projector 最后 `return self.ln(out)` | 与 text embedding 尺度对齐，缓解模式塌缩 |

### 2.5 推理与评估时的对齐

| 策略 | 实现方式 | 作用 |
|------|----------|------|
| **与训练一致的 prompt** | eval 使用相同 prompt + apply_chat_template(..., add_generation_prompt=True) | 生成时「3D + user + assistant 开头」与训练分布一致 |
| **同一 truncate_mode** | 训练用 random_sample 时，eval 也可配 truncate_mode=random_sample | 推理与训练截断逻辑一致，避免分布偏移 |

---

## 三、可行性评估

### 3.1 设计是否合理（能否对齐）

- **接口一致**：3D 通过 Projector 映射到 VL 的 hidden_size，与文本 embedding 同一空间，再以 prefix 形式输入，这是成熟的多模态做法，**接口上可以正常对齐**。
- **监督明确**：只对 caption 做 LM loss，梯度经 VL 回传到 Projector（以及 LoRA），**目标清晰，能驱动「3D→语义」的学习**。
- **容量与稳定性**：Projector 可 1 层或多层 + 3D 位置编码 + LayerNorm；VAE 冻结保证 3D 表示稳定；LoRA 可选，在不过度动 VL 的前提下适配 3D 前缀，**结构上具备对齐能力**。

结论：**从策略和结构上看，可以正常进行对齐**；是否收敛、效果好坏取决于数据、超参和算力。

### 3.2 可能风险与依赖

| 风险/依赖 | 说明 | 建议 |
|-----------|------|------|
| **3D 序列过长** | max_3d_tokens=8192 时，3D+文本可能接近或超过 VL 的 max position | 监控总长度；必要时减小 max_3d_tokens 或使用 random_sample 截断 |
| **Caption 质量与多样性** | 标注单一或噪声大会限制「理解 3D」的上限 | 清洗/扩充 caption，或做多描述增强 |
| **Projector 过弱** | 单层 Linear 可能表达不足 | 配置中已用 projector_layers=3；可按需尝试更深或 3D 位置编码 |
| **VL 完全忽略 3D** | 若 3D 前缀与文本分布差异大，VL 可能主要依赖 prompt | 开 LoRA 让 VL 适配 3D；保证足够步数/epoch |
| **数据量** | 对齐效果依赖 3D–caption 对的数量与多样性 | max_samples=0 表示全量；小数据时可适当加大 epoch、减小 batch、监控过拟合 |

### 3.3 与常见多模态对齐的对比

- **与 LLaVA/Shikra 等**：都是「另一模态 → 线性/MLP 映射到 LLM 隐藏空间 → 与文本拼接做 causal LM」，本项目的 3D 分支与之一致，**范式可行**。
- **与纯 contrastive 不同**：这里没有单独的 contrastive loss，仅靠「3D prefix + 生成 caption」的 LM 训练做对齐，**依赖生成质量和数据**，但实现简单、易复现。

---

## 四、小结

- **策略与结构**：当前对齐策略（3D 码本 → 序列化 → Projector → 与文本拼接 → 仅监督 caption 的 LM + 可选 LoRA）与模型结构（冻结 VAE、可训 Projector、可选 LoRA）**设计完整、接口正确**。
- **可行性**：**能够正常进行对齐**；效果取决于数据质量与规模、prompt、max_3d_tokens、Projector 深度、LoRA 与学习率等。建议先用小规模数据与烟雾测试确认 loss 下降与生成合理，再放大数据和算力。
