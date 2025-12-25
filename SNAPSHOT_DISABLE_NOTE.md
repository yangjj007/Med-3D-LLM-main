# 快照功能禁用说明

## 问题描述

在训练CT VQVAE模型时，快照（snapshot）功能会导致浮点异常崩溃：
```
Floating point exception (core dumped)
```

这个问题发生在模型编码阶段（`vqvae.Encode(x)`），可能是由于：
1. SpConv库或CUDA kernel中的数值问题
2. 处理大量稀疏点（100k-500k）时的内存或计算溢出
3. 特定的硬件/驱动组合问题

## 解决方案

已在配置文件中添加 `disable_snapshot` 参数来禁用快照功能：

### 配置文件修改

在 `configs/vae/ct_vqvae_stage1.json` 中：

```json
"trainer": {
    "name": "SparseSDF_VQVAETrainer",
    "args": {
        ...
        "disable_snapshot": true,
        ...
    }
}
```

### 影响

**禁用快照后：**
- ✅ 训练可以正常进行
- ✅ 模型检查点仍会正常保存（每5000步）
- ✅ 训练日志仍会正常记录
- ❌ 不会生成可视化图像（用于监控训练效果）
- ❌ 无法在训练过程中直观查看重建质量

**快照功能的作用：**
- 定期从数据集采样
- 通过模型进行编码和解码
- 将3D稀疏数据可视化为2D切片图像
- 保存到 `outputs/.../samples/` 目录

## 替代方案

如果需要监控训练效果，可以：

1. **使用TensorBoard日志**：查看损失曲线
   ```bash
   tensorboard --logdir outputs/ct_vqvae_lung_stage1
   ```

2. **训练后手动测试**：训练完成后，编写单独的推理脚本测试模型

3. **减少点数**（已尝试但未解决）：
   - 在 `run_snapshot` 中将 `max_points` 限制为 100000
   - 问题仍然存在

## 未来调试方向

如果需要修复快照功能，可以尝试：

1. **更新SpConv库**：可能是库的bug
2. **检查CUDA版本兼容性**
3. **在CPU上运行快照**（会很慢）
4. **使用更小的模型分辨率进行快照**
5. **分批处理编码**：将大的SparseTensor分成多个小块

## 如何重新启用快照

如果将来想重新启用快照功能，只需：

1. 在配置文件中设置：
   ```json
   "disable_snapshot": false
   ```
   或直接删除该行（默认为false）

2. 重新运行训练命令

---

**创建日期**: 2025-12-25
**相关文件**: 
- `trellis/trainers/base.py`
- `trellis/trainers/vae/sparse_sdf_vqvae.py`
- `configs/vae/ct_vqvae_stage1.json`

