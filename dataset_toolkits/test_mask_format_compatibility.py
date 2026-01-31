"""
测试掩码格式兼容性

验证新的掩码处理逻辑能够正确处理以下格式：
1. 单通道二值掩码 (4D, 第一维=1): (1, H, W, D)
2. 单通道标签掩码 (3D): (H, W, D) 标签值0,1,2...
3. one-hot编码 (4D, 互斥): (C, H, W, D) 每个通道互斥
4. 多通道布尔掩码 (4D, 可能重叠): (C, H, W, D) 通道间可能重叠
"""

import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_toolkits.process_m3d_seg_format import (
    detect_mask_format,
    parse_multi_channel_mask
)


def test_single_channel_4d():
    """测试单通道二值掩码 (4D, 第一维=1)"""
    print("\n" + "="*70)
    print("测试1: 单通道二值掩码 (1, 512, 512, 221)")
    print("="*70)
    
    # 模拟PANCREAS_0004的数据
    mask = np.zeros((1, 512, 512, 221), dtype=np.uint8)
    # 添加一些标注数据
    mask[0, 200:300, 200:300, 100:120] = 1
    
    print(f"输入形状: {mask.shape}")
    print(f"唯一值: {np.unique(mask)}")
    print(f"标注体素数: {mask.sum()}")
    
    # 检测格式
    mask_type, num_channels = detect_mask_format(mask)
    print(f"检测结果: {mask_type}, 通道数: {num_channels}")
    
    # 解析（不进行分辨率适配，使用原始分辨率）
    seg_adapted, seg_mask_type, seg_num_channels = parse_multi_channel_mask(mask, target_resolution=512)
    print(f"适配后形状: {seg_adapted.shape}")
    print(f"适配后维度: {seg_adapted.ndim}")
    print(f"标注保留: {seg_adapted.sum()} 体素")
    
    # 验证
    assert seg_adapted.ndim == 3, "应该降维到3D"
    assert seg_adapted.sum() > 0, "标注数据应该保留"
    print("✓ 测试通过：单通道二值掩码正确处理")


def test_single_label_3d():
    """测试单通道标签掩码 (3D)"""
    print("\n" + "="*70)
    print("测试2: 单通道标签掩码 (512, 512, 221)")
    print("="*70)
    
    # 模拟3D标签数据
    mask = np.zeros((512, 512, 221), dtype=np.uint8)
    # 添加多个器官标签
    mask[100:200, 100:200, 50:100] = 1  # 肝脏
    mask[250:350, 250:350, 100:150] = 2  # 肾脏
    mask[400:450, 400:450, 150:180] = 3  # 脾脏
    
    print(f"输入形状: {mask.shape}")
    print(f"唯一值: {np.unique(mask)}")
    print(f"标签分布:")
    for label_id in np.unique(mask):
        count = (mask == label_id).sum()
        print(f"  标签 {label_id}: {count} 体素")
    
    # 检测格式
    mask_type, num_channels = detect_mask_format(mask)
    print(f"检测结果: {mask_type}, 通道数: {num_channels}")
    
    # 解析
    seg_adapted, seg_mask_type, seg_num_channels = parse_multi_channel_mask(mask, target_resolution=512)
    print(f"适配后形状: {seg_adapted.shape}")
    print(f"适配后标签: {np.unique(seg_adapted)}")
    
    # 验证
    assert seg_adapted.ndim == 3, "应该保持3D"
    assert len(np.unique(seg_adapted)) == 4, "应该有4个标签（包括背景）"
    for label_id in [1, 2, 3]:
        assert (seg_adapted == label_id).sum() > 0, f"标签{label_id}数据应该保留"
    print("✓ 测试通过：单通道标签掩码正确处理")


def test_one_hot_encoding():
    """测试one-hot编码 (4D, 互斥)"""
    print("\n" + "="*70)
    print("测试3: one-hot编码 (3, 256, 256, 128)")
    print("="*70)
    
    # 模拟one-hot编码数据（3个器官，互斥）
    mask = np.zeros((3, 256, 256, 128), dtype=np.uint8)
    mask[0, 50:100, 50:100, 30:60] = 1   # 器官1
    mask[1, 120:180, 120:180, 60:90] = 1  # 器官2
    mask[2, 200:230, 200:230, 90:110] = 1  # 器官3
    
    print(f"输入形状: {mask.shape}")
    print(f"通道重叠检查:")
    overlap = mask.sum(axis=0).max()
    print(f"  最大重叠值: {overlap} (应该=1，表示无重叠)")
    
    for ch in range(3):
        print(f"  通道 {ch}: {mask[ch].sum()} 体素")
    
    # 检测格式
    mask_type, num_channels = detect_mask_format(mask)
    print(f"检测结果: {mask_type}, 通道数: {num_channels}")
    
    # 解析
    seg_adapted, seg_mask_type, seg_num_channels = parse_multi_channel_mask(mask, target_resolution=256)
    print(f"适配后形状: {seg_adapted.shape}")
    print(f"适配后标签: {np.unique(seg_adapted)}")
    
    # 验证
    assert seg_adapted.ndim == 3, "one-hot应该转换为3D单标签"
    assert mask_type == 'one_hot', "应该检测为one-hot编码"
    # 验证每个器官的体素都被保留
    for label_id in [1, 2, 3]:
        assert (seg_adapted == label_id).sum() > 0, f"器官{label_id}数据应该保留"
    print("✓ 测试通过：one-hot编码正确处理")


def test_multi_channel_with_overlap():
    """测试多通道布尔掩码（有重叠）"""
    print("\n" + "="*70)
    print("测试4: 多通道布尔掩码（有重叠） (30, 256, 256, 128)")
    print("="*70)
    
    # 模拟case_01的数据（30个通道，可能重叠）
    mask = np.zeros((30, 256, 256, 128), dtype=bool)
    
    # 添加器官数据，故意让某些区域重叠（如肝脏和肿瘤）
    mask[0, 50:150, 50:150, 30:80] = True   # 肝脏（大）
    mask[1, 100:130, 100:130, 50:70] = True  # 肝肿瘤（与肝脏重叠）
    mask[2, 180:220, 180:220, 80:110] = True  # 肾脏
    mask[3, 200:210, 200:210, 90:100] = True  # 肾肿瘤（与肾脏重叠）
    
    # 再添加一些非重叠器官
    for i in range(4, 10):
        x = (i % 4) * 60
        y = (i % 4) * 60
        z = (i % 3) * 40
        mask[i, x:x+40, y:y+40, z:z+30] = True
    
    print(f"输入形状: {mask.shape}")
    print(f"通道重叠检查:")
    overlap = mask.sum(axis=0).max()
    print(f"  最大重叠值: {overlap} (>1表示有重叠)")
    
    # 统计非空通道
    non_empty_channels = 0
    for ch in range(mask.shape[0]):
        count = mask[ch].sum()
        if count > 0:
            non_empty_channels += 1
            if ch < 5:  # 只显示前5个
                print(f"  通道 {ch}: {count} 体素")
    print(f"  ... (共 {non_empty_channels} 个非空通道)")
    
    # 检测格式
    mask_type, num_channels = detect_mask_format(mask)
    print(f"检测结果: {mask_type}, 通道数: {num_channels}")
    
    # 解析
    seg_adapted, seg_mask_type, seg_num_channels = parse_multi_channel_mask(mask, target_resolution=256)
    print(f"适配后形状: {seg_adapted.shape}")
    print(f"适配后维度: {seg_adapted.ndim}")
    
    # 验证
    assert seg_adapted.ndim == 4, "多通道掩码应该保持4D结构"
    assert mask_type == 'multi_channel', "应该检测为多通道布尔掩码"
    assert seg_adapted.shape[0] == 30, "应该保持30个通道"
    
    # 验证重叠区域的数据都被保留（关键测试！）
    print("\n验证重叠区域数据保留:")
    # 检查肝脏和肝肿瘤的重叠区域
    liver = seg_adapted[0].astype(bool)
    liver_tumor = seg_adapted[1].astype(bool)
    overlap_region = liver & liver_tumor
    overlap_count = overlap_region.sum()
    print(f"  肝脏 (通道0): {liver.sum()} 体素")
    print(f"  肝肿瘤 (通道1): {liver_tumor.sum()} 体素")
    print(f"  重叠区域: {overlap_count} 体素")
    
    assert overlap_count > 0, "重叠区域应该被保留！"
    print("  ✓ 重叠区域数据完整保留！")
    
    # 验证所有非空通道都被保留
    for ch in range(mask.shape[0]):
        original_count = mask[ch].sum()
        adapted_count = seg_adapted[ch].sum()
        if original_count > 0:
            # 分辨率适配可能导致数量略有变化，但不应该完全丢失
            assert adapted_count > 0, f"通道{ch}的数据不应该丢失"
    
    print("✓ 测试通过：多通道布尔掩码正确处理，重叠数据完整保留")


def test_edge_case_large_channels():
    """测试边界情况：大量通道"""
    print("\n" + "="*70)
    print("测试5: 边界情况 - 大量通道 (50, 128, 128, 64)")
    print("="*70)
    
    # 模拟50个通道的掩码（测试之前的20通道分界线已被移除）
    mask = np.zeros((50, 128, 128, 64), dtype=bool)
    
    # 随机添加一些器官数据
    for i in range(0, 50, 5):
        x = (i % 5) * 25
        y = (i % 5) * 25
        z = (i % 4) * 15
        mask[i, x:x+20, y:y+20, z:z+10] = True
    
    print(f"输入形状: {mask.shape}")
    print(f"非空通道数: {sum(1 for ch in range(50) if mask[ch].sum() > 0)}")
    
    # 检测格式
    mask_type, num_channels = detect_mask_format(mask)
    print(f"检测结果: {mask_type}, 通道数: {num_channels}")
    
    # 解析
    seg_adapted, seg_mask_type, seg_num_channels = parse_multi_channel_mask(mask, target_resolution=128)
    print(f"适配后形状: {seg_adapted.shape}")
    
    # 验证
    assert seg_adapted.ndim == 4, "应该保持4D结构（移除了20通道限制）"
    assert seg_adapted.shape[0] == 50, "应该保持50个通道"
    print("✓ 测试通过：大量通道正确处理（20通道分界线已移除）")


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("掩码格式兼容性测试")
    print("="*70)
    
    try:
        test_single_channel_4d()
        test_single_label_3d()
        test_one_hot_encoding()
        test_multi_channel_with_overlap()
        test_edge_case_large_channels()
        
        print("\n" + "="*70)
        print("✓ 所有测试通过！")
        print("="*70)
        print("\n总结:")
        print("1. ✓ 单通道二值掩码 (4D, shape[0]=1) - 正确降维到3D")
        print("2. ✓ 单通道标签掩码 (3D) - 保持3D结构")
        print("3. ✓ one-hot编码 (4D, 互斥) - 转换为3D单标签")
        print("4. ✓ 多通道布尔掩码 (4D, 重叠) - 保持4D，数据完整")
        print("5. ✓ 大量通道 (50通道) - 正确处理，移除20通道限制")
        print("\n核心改进:")
        print("- 移除了20通道分界线的任意限制")
        print("- 多通道掩码保持4D结构，避免数据丢失")
        print("- 重叠区域的器官标注完整保留")
        print("="*70)
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ 测试错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

