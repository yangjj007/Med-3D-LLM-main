#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试模型导入是否正常
"""

print("正在测试模型导入...")

# 测试 1: 导入新类名
try:
    from trellis.models import SparseSDFVQVAE
    print("✅ 成功导入 SparseSDFVQVAE")
except Exception as e:
    print(f"❌ 导入 SparseSDFVQVAE 失败: {e}")

# 测试 2: 导入旧类名（别名）
try:
    from trellis.models import Direct3DS2_VQVAE
    print("✅ 成功导入 Direct3DS2_VQVAE (别名)")
except Exception as e:
    print(f"❌ 导入 Direct3DS2_VQVAE 失败: {e}")

# 测试 3: 验证两者是同一个类
try:
    from trellis.models import SparseSDFVQVAE, Direct3DS2_VQVAE
    if SparseSDFVQVAE is Direct3DS2_VQVAE:
        print("✅ SparseSDFVQVAE 和 Direct3DS2_VQVAE 是同一个类（别名工作正常）")
    else:
        print("⚠️ SparseSDFVQVAE 和 Direct3DS2_VQVAE 不是同一个类")
except Exception as e:
    print(f"❌ 验证失败: {e}")

# 测试 4: 使用 getattr 方式导入（与 train.py 相同）
try:
    import trellis.models as models
    model_class = getattr(models, 'Direct3DS2_VQVAE')
    print(f"✅ 使用 getattr 成功获取 Direct3DS2_VQVAE: {model_class}")
except Exception as e:
    print(f"❌ 使用 getattr 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成！")

