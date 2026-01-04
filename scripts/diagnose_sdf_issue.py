#!/usr/bin/env python
"""
SDF 功能诊断脚本

检查 SDF 计算所需的依赖和配置是否正确
"""

import sys
import os

print("=" * 80)
print("SDF 功能诊断")
print("=" * 80)
print()

# 检查1: PyTorch 和 CUDA
print("1. 检查 PyTorch 和 CUDA")
print("-" * 40)
try:
    import torch
    print(f"✓ PyTorch 已安装: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用")
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("✗ CUDA 不可用")
        print("  警告: SDF 计算需要 CUDA 支持")
except ImportError as e:
    print(f"✗ PyTorch 未安装: {e}")
    print("  请安装: pip install torch")
print()

# 检查2: TRELLIS
print("2. 检查 TRELLIS")
print("-" * 40)
try:
    from trellis.utils.mesh_utils import dense_voxel_to_sparse_sdf
    print("✓ TRELLIS 可用")
    print("  可以导入 dense_voxel_to_sparse_sdf 函数")
except ImportError as e:
    print(f"✗ TRELLIS 不可用: {e}")
    print("  这是导致 SDF 计算失败的主要原因！")
    print("  请检查:")
    print("  1. TRELLIS 是否已安装")
    print("  2. trellis/utils/mesh_utils.py 文件是否存在")
    print("  3. dense_voxel_to_sparse_sdf 函数是否已实现")
except Exception as e:
    print(f"✗ TRELLIS 导入错误: {e}")
print()

# 检查3: CT 预处理模块
print("3. 检查 CT 预处理模块")
print("-" * 40)
try:
    from dataset_toolkits.ct_preprocessing import (
        check_cuda_available,
        check_trellis_available,
        convert_window_to_sdf
    )
    print("✓ CT 预处理模块导入成功")
    
    cuda_ok = check_cuda_available()
    trellis_ok = check_trellis_available()
    
    print(f"  CUDA 检查: {'✓' if cuda_ok else '✗'}")
    print(f"  TRELLIS 检查: {'✓' if trellis_ok else '✗'}")
    
    if cuda_ok and trellis_ok:
        print("\n  ✓ 所有依赖都满足，SDF 计算应该可以工作")
    else:
        print("\n  ✗ 缺少必要的依赖，SDF 计算将被跳过")
        
except ImportError as e:
    print(f"✗ CT 预处理模块导入失败: {e}")
print()

# 检查4: 测试 SDF 转换
print("4. 测试 SDF 转换功能")
print("-" * 40)
try:
    from dataset_toolkits.ct_preprocessing import (
        check_cuda_available,
        check_trellis_available,
        convert_window_to_sdf
    )
    import numpy as np
    
    if check_cuda_available() and check_trellis_available():
        # 创建测试数据
        test_data = np.zeros((64, 64, 64), dtype=np.uint8)
        test_data[16:48, 16:48, 16:48] = 1  # 中心立方体
        
        print("  创建测试数据: 64x64x64 立方体")
        print(f"  正值体素数: {np.sum(test_data)}")
        
        try:
            result = convert_window_to_sdf(test_data, resolution=64)
            print(f"  ✓ SDF 转换成功")
            print(f"  稀疏点数: {len(result['sparse_index'])}")
            print(f"  分辨率: {result['resolution']}")
        except Exception as e:
            print(f"  ✗ SDF 转换失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  跳过测试 (依赖不满足)")
        
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
print()

# 总结
print("=" * 80)
print("诊断总结")
print("=" * 80)

# 重新检查所有条件
all_ok = True
messages = []

try:
    import torch
    if not torch.cuda.is_available():
        all_ok = False
        messages.append("CUDA 不可用")
except:
    all_ok = False
    messages.append("PyTorch 未安装")

try:
    from trellis.utils.mesh_utils import dense_voxel_to_sparse_sdf
except:
    all_ok = False
    messages.append("TRELLIS 不可用或 dense_voxel_to_sparse_sdf 函数未实现")

if all_ok:
    print("✓ 所有检查通过！SDF 计算功能应该可以正常工作")
    print()
    print("如果仍然无法生成 .npz 文件，请检查:")
    print("1. 运行命令时是否看到警告信息")
    print("2. 二值化数据是否过于稀疏 (< 100 个正值体素)")
    print("3. 查看详细的错误日志")
else:
    print("✗ 检查未通过，SDF 计算功能无法工作")
    print()
    print("问题:")
    for msg in messages:
        print(f"  - {msg}")
    print()
    print("解决方案:")
    if "CUDA 不可用" in str(messages):
        print("  1. 安装支持 CUDA 的 PyTorch 版本")
        print("     访问: https://pytorch.org/get-started/locally/")
    if "TRELLIS" in str(messages):
        print("  2. 确保 TRELLIS 已正确安装")
        print("  3. 检查 trellis/utils/mesh_utils.py 是否实现了 dense_voxel_to_sparse_sdf")
        print("     如果没有，需要实现这个函数或使用替代方案")

print("=" * 80)

