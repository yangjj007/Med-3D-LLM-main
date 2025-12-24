"""
CT预处理模块单元测试

测试所有核心功能模块的正确性。
"""

import sys
import os
import numpy as np
import tempfile
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入要测试的模块
from dataset_toolkits.ct_preprocessing import (
    # Config
    WINDOW_CONFIGS,
    AIR_HU_VALUE,
    DEFAULT_RESOLUTION,
    get_window_config,
    get_window_for_organ,
    
    # Resolution adapter
    determine_target_resolution,
    adapt_resolution,
    check_resolution_compatibility,
    
    # Window processor
    apply_window_binarization,
    process_all_windows,
    get_window_filename,
    
    # Organ extractor
    extract_organ_mask,
    extract_organ_with_window,
    validate_segmentation,
    compute_organ_statistics,
)


def test_config():
    """测试配置模块"""
    print("=" * 50)
    print("测试1: 配置模块")
    print("=" * 50)
    
    # 测试窗口配置
    assert 'lung' in WINDOW_CONFIGS
    assert 'bone' in WINDOW_CONFIGS
    assert 'soft_tissue' in WINDOW_CONFIGS
    assert 'brain' in WINDOW_CONFIGS
    print("✓ 窗口配置完整")
    
    # 测试获取窗口配置
    lung_config = get_window_config('lung')
    assert lung_config['window_level'] == -600
    assert lung_config['window_width'] == 1500
    print("✓ 窗口配置获取正确")
    
    # 测试器官窗口映射
    liver_window = get_window_for_organ('liver')
    assert liver_window == 'soft_tissue'
    print("✓ 器官窗口映射正确")
    
    # 测试空气HU值
    assert AIR_HU_VALUE == -1000
    print("✓ 空气HU值正确")
    
    # 测试默认分辨率
    assert DEFAULT_RESOLUTION == 512
    print("✓ 默认分辨率正确")
    
    print("✓ 配置模块测试通过\n")


def test_resolution_adapter():
    """测试分辨率适配器"""
    print("=" * 50)
    print("测试2: 分辨率适配器")
    print("=" * 50)
    
    # 测试1: 512x512x100 -> 512³
    test_array_1 = np.random.randn(512, 512, 100).astype(np.float32)
    target_res_1 = determine_target_resolution(test_array_1.shape)
    assert target_res_1 == 512
    print(f"✓ {test_array_1.shape} -> {target_res_1}³")
    
    adapted_1 = adapt_resolution(test_array_1, target_res_1)
    assert adapted_1.shape == (512, 512, 512)
    print(f"✓ 适配后形状正确: {adapted_1.shape}")
    
    # 测试2: 256x256x150 -> 512³
    test_array_2 = np.random.randn(256, 256, 150).astype(np.float32)
    target_res_2 = determine_target_resolution(test_array_2.shape)
    assert target_res_2 == 512
    adapted_2 = adapt_resolution(test_array_2, target_res_2)
    assert adapted_2.shape == (512, 512, 512)
    print(f"✓ {test_array_2.shape} -> {adapted_2.shape}")
    
    # 测试3: 1024x1024x200 -> 1024³
    test_array_3 = np.random.randn(600, 600, 200).astype(np.float32)
    target_res_3 = determine_target_resolution(test_array_3.shape)
    assert target_res_3 == 1024
    adapted_3 = adapt_resolution(test_array_3, target_res_3)
    assert adapted_3.shape == (1024, 1024, 1024)
    print(f"✓ {test_array_3.shape} -> {adapted_3.shape}")
    
    # 测试4: 带通道的数组
    test_array_4 = np.random.randn(1, 512, 512, 100).astype(np.float32)
    adapted_4 = adapt_resolution(test_array_4, 512)
    assert adapted_4.shape == (1, 512, 512, 512)
    print(f"✓ 带通道数组: {test_array_4.shape} -> {adapted_4.shape}")
    
    # 测试5: 检查兼容性
    assert check_resolution_compatibility((512, 512, 100), 512) == True
    assert check_resolution_compatibility((1025, 512, 100), 1024) == False
    print("✓ 分辨率兼容性检查正确")
    
    # 测试6: 填充值检查
    test_array_5 = np.ones((100, 100, 100), dtype=np.float32)
    adapted_5 = adapt_resolution(test_array_5, 512, fill_value=-1000)
    # 检查填充区域的值
    assert np.all(adapted_5[100:, :, :] == -1000)
    print("✓ 填充值正确")
    
    print("✓ 分辨率适配器测试通过\n")


def test_window_processor():
    """测试窗口处理器"""
    print("=" * 50)
    print("测试3: 窗口处理器")
    print("=" * 50)
    
    # 创建测试CT数据
    test_ct = np.zeros((100, 100, 100), dtype=np.float32)
    test_ct[20:40, 20:40, 20:40] = -600  # 肺组织
    test_ct[40:60, 40:60, 40:60] = 50    # 软组织
    test_ct[60:80, 60:80, 60:80] = 300   # 骨组织
    
    # 测试1: 单个窗口二值化
    lung_binary = apply_window_binarization(test_ct, -600, 1500)
    assert lung_binary.shape == test_ct.shape
    assert lung_binary.dtype == np.uint8
    assert np.sum(lung_binary) > 0
    print(f"✓ 肺窗二值化: {np.sum(lung_binary)} 个正值体素")
    
    # 测试2: 所有窗口处理
    all_windows = process_all_windows(test_ct, binarize=True)
    assert len(all_windows) >= 4
    assert 'lung' in all_windows
    assert 'bone' in all_windows
    assert 'soft_tissue' in all_windows
    print(f"✓ 处理了 {len(all_windows)} 个窗口")
    
    # 测试3: 文件名生成
    lung_filename = get_window_filename('lung')
    assert 'lung' in lung_filename
    assert 'w1500' in lung_filename
    assert 'l-600' in lung_filename
    print(f"✓ 文件名生成: {lung_filename}")
    
    # 测试4: 窗口范围检查
    bone_binary = all_windows['bone']
    # 骨窗应该能捕获骨组织（HU=300）
    bone_region = bone_binary[60:80, 60:80, 60:80]
    assert np.sum(bone_region) > 0
    print(f"✓ 窗口范围正确")
    
    print("✓ 窗口处理器测试通过\n")


def test_organ_extractor():
    """测试器官提取器"""
    print("=" * 50)
    print("测试4: 器官提取器")
    print("=" * 50)
    
    # 创建测试数据
    ct = np.random.randn(100, 100, 100).astype(np.float32) * 100
    segmentation = np.zeros((100, 100, 100), dtype=np.int32)
    
    # 添加器官
    segmentation[20:40, 20:40, 20:40] = 1  # 肝脏
    segmentation[50:70, 50:70, 50:70] = 2  # 肾脏
    ct[20:40, 20:40, 20:40] = 50           # 软组织HU值
    ct[50:70, 50:70, 50:70] = 30           # 软组织HU值
    
    # 测试1: 提取器官掩码
    liver_mask = extract_organ_mask(segmentation, organ_label=1)
    assert liver_mask.shape == segmentation.shape
    assert liver_mask.dtype == np.uint8
    expected_voxels = 20 * 20 * 20
    assert np.sum(liver_mask) == expected_voxels
    print(f"✓ 器官掩码提取: {np.sum(liver_mask)} 体素")
    
    # 测试2: 器官窗口提取
    soft_tissue_config = get_window_config('soft_tissue')
    liver_window = extract_organ_with_window(
        ct, segmentation, organ_label=1, window_config=soft_tissue_config
    )
    assert liver_window.shape == ct.shape
    assert np.sum(liver_window) > 0
    print(f"✓ 器官窗口提取: {np.sum(liver_window)} 体素")
    
    # 测试3: 器官统计
    liver_stats = compute_organ_statistics(ct, segmentation, organ_label=1)
    assert liver_stats['organ_label'] == 1
    assert liver_stats['voxel_count'] == expected_voxels
    assert 'hu_mean' in liver_stats
    assert 'hu_std' in liver_stats
    print(f"✓ 器官统计: 平均HU={liver_stats['hu_mean']:.2f}")
    
    # 测试4: 分割验证
    is_valid, message = validate_segmentation(segmentation, ct)
    assert is_valid == True
    print(f"✓ 分割验证: {message}")
    
    # 测试5: 空分割（应该失败）
    empty_seg = np.zeros_like(segmentation)
    is_valid_empty, _ = validate_segmentation(empty_seg, ct)
    assert is_valid_empty == False
    print(f"✓ 空分割检测正确")
    
    print("✓ 器官提取器测试通过\n")


def test_integration():
    """集成测试"""
    print("=" * 50)
    print("测试5: 集成测试")
    print("=" * 50)
    
    # 创建完整的测试场景
    print("创建测试数据...")
    
    # 1. 不规则CT（512x512x100）
    ct_irregular = np.random.randn(512, 512, 100).astype(np.float32) * 100 + 50
    # 添加一些典型的HU值区域
    ct_irregular[100:200, 100:200, 10:30] = -600  # 肺
    ct_irregular[200:300, 200:300, 40:60] = 300   # 骨
    ct_irregular[300:400, 300:400, 70:90] = 50    # 软组织
    
    print(f"原始CT形状: {ct_irregular.shape}")
    
    # 2. 分辨率适配
    target_res = determine_target_resolution(ct_irregular.shape)
    ct_adapted = adapt_resolution(ct_irregular, target_res)
    print(f"适配后形状: {ct_adapted.shape} (目标: {target_res}³)")
    assert ct_adapted.shape == (target_res, target_res, target_res)
    
    # 3. 窗口处理
    windows = process_all_windows(ct_adapted[:512, :512, :100], binarize=True)
    print(f"处理了 {len(windows)} 个窗口")
    
    # 4. 创建分割标签
    segmentation = np.zeros((512, 512, 100), dtype=np.int32)
    segmentation[100:200, 100:200, 10:30] = 1  # 肺
    segmentation[300:400, 300:400, 70:90] = 2  # 肝脏
    seg_adapted = adapt_resolution(segmentation, target_res, fill_value=0, mode='constant')
    
    # 5. 器官提取
    liver_mask = extract_organ_mask(seg_adapted, organ_label=2)
    liver_voxels = np.sum(liver_mask)
    print(f"肝脏体素数: {liver_voxels}")
    
    # 6. 器官窗口
    soft_tissue_config = get_window_config('soft_tissue')
    liver_window = extract_organ_with_window(
        ct_adapted, seg_adapted, organ_label=2, window_config=soft_tissue_config
    )
    liver_window_voxels = np.sum(liver_window)
    print(f"肝脏窗口体素数: {liver_window_voxels}")
    
    # 验证
    assert liver_window_voxels <= liver_voxels  # 窗口应该不多于原始掩码
    
    print("✓ 集成测试通过\n")


def test_file_operations():
    """测试文件操作"""
    print("=" * 50)
    print("测试6: 文件操作")
    print("=" * 50)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"临时目录: {tmpdir}")
        
        # 1. 测试保存和加载NPY文件
        test_array = np.random.randn(100, 100, 100).astype(np.float32)
        npy_path = os.path.join(tmpdir, 'test_ct.npy')
        np.save(npy_path, test_array)
        
        loaded_array = np.load(npy_path)
        assert np.allclose(test_array, loaded_array)
        print("✓ NPY文件保存/加载正确")
        
        # 2. 测试JSON配置
        test_config = {
            "dataset_name": "Test",
            "organ_labels": {
                "1": {"name": "liver", "window": "soft_tissue"}
            }
        }
        json_path = os.path.join(tmpdir, 'config.json')
        with open(json_path, 'w') as f:
            json.dump(test_config, f)
        
        with open(json_path, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config['dataset_name'] == "Test"
        print("✓ JSON配置保存/加载正确")
        
        # 3. 测试目录结构创建
        output_structure = os.path.join(tmpdir, 'processed', 'case_001')
        os.makedirs(output_structure, exist_ok=True)
        os.makedirs(os.path.join(output_structure, 'windows'), exist_ok=True)
        os.makedirs(os.path.join(output_structure, 'organs'), exist_ok=True)
        os.makedirs(os.path.join(output_structure, 'masks'), exist_ok=True)
        
        assert os.path.exists(output_structure)
        assert os.path.exists(os.path.join(output_structure, 'windows'))
        print("✓ 目录结构创建正确")
    
    print("✓ 文件操作测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("=" * 50)
    print("   CT预处理模块单元测试")
    print("=" * 50)
    print("\n")
    
    try:
        test_config()
        test_resolution_adapter()
        test_window_processor()
        test_organ_extractor()
        test_integration()
        test_file_operations()
        
        print("=" * 50)
        print("   所有测试通过！✓")
        print("=" * 50)
        print("\n")
        return True
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ 测试错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

