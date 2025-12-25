"""
Test script for CT Window Sparse Dataset

Tests the CTWindowSparseSDF dataset to verify:
1. Data loading works correctly
2. Sparse conversion is successful
3. Batch collation works properly
4. Window types are correctly loaded
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from trellis.datasets import CTWindowSparseSDF


def test_basic_loading(data_root: str, window_type: str = 'lung'):
    """
    Test basic dataset loading functionality.
    
    Args:
        data_root: Root directory containing processed CT data
        window_type: Window type to test
    """
    print("\n" + "=" * 80)
    print("Test 1: Basic Dataset Loading")
    print("=" * 80)
    
    try:
        dataset = CTWindowSparseSDF(
            roots=data_root,
            resolution=512,
            window_type=window_type,
            min_points=100,
            max_points=500000
        )
        
        print(f"\n✓ Dataset created successfully")
        print(f"  Total instances: {len(dataset)}")
        print(f"  Window type: {window_type}")
        
        if len(dataset) == 0:
            print("\n✗ Error: Dataset is empty. Please check your data directory.")
            return False
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_sample(dataset: CTWindowSparseSDF):
    """
    Test loading a single sample.
    
    Args:
        dataset: CTWindowSparseSDF instance
    """
    print("\n" + "=" * 80)
    print("Test 2: Single Sample Loading")
    print("=" * 80)
    
    try:
        sample = dataset[0]
        
        print(f"\n✓ Sample loaded successfully")
        print(f"  Sparse SDF shape: {sample['sparse_sdf'].shape}")
        print(f"  Sparse index shape: {sample['sparse_index'].shape}")
        print(f"  SDF value range: [{sample['sparse_sdf'].min():.3f}, {sample['sparse_sdf'].max():.3f}]")
        print(f"  Coordinate range:")
        print(f"    X: [{sample['sparse_index'][:, 0].min()}, {sample['sparse_index'][:, 0].max()}]")
        print(f"    Y: [{sample['sparse_index'][:, 1].min()}, {sample['sparse_index'][:, 1].max()}]")
        print(f"    Z: [{sample['sparse_index'][:, 2].min()}, {sample['sparse_index'][:, 2].max()}]")
        
        # Verify data types
        assert sample['sparse_sdf'].dtype == torch.float32, "SDF values should be float32"
        assert sample['sparse_index'].dtype == torch.int64, "Indices should be int64"
        
        # Verify shapes
        N = sample['sparse_sdf'].shape[0]
        assert sample['sparse_sdf'].shape == (N, 1), "SDF should be [N, 1]"
        assert sample['sparse_index'].shape == (N, 3), "Index should be [N, 3]"
        
        print(f"\n✓ All assertions passed")
        return True
    
    except Exception as e:
        print(f"\n✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_collation(dataset: CTWindowSparseSDF, batch_size: int = 2):
    """
    Test batch collation functionality.
    
    Args:
        dataset: CTWindowSparseSDF instance
        batch_size: Batch size to test
    """
    print("\n" + "=" * 80)
    print(f"Test 3: Batch Collation (batch_size={batch_size})")
    print("=" * 80)
    
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn
        )
        
        batch = next(iter(dataloader))
        
        print(f"\n✓ Batch collated successfully")
        print(f"  Sparse SDF shape: {batch['sparse_sdf'].shape}")
        print(f"  Sparse index shape: {batch['sparse_index'].shape}")
        print(f"  Batch idx shape: {batch['batch_idx'].shape}")
        print(f"  Unique batch indices: {torch.unique(batch['batch_idx']).tolist()}")
        
        # Verify batch indices
        unique_batches = torch.unique(batch['batch_idx'])
        assert len(unique_batches) <= batch_size, "Number of batches should not exceed batch_size"
        
        # Verify shapes
        total_N = batch['sparse_sdf'].shape[0]
        assert batch['sparse_sdf'].shape == (total_N, 1), "Batch SDF should be [total_N, 1]"
        assert batch['sparse_index'].shape == (total_N, 3), "Batch index should be [total_N, 3]"
        assert batch['batch_idx'].shape == (total_N,), "Batch idx should be [total_N]"
        
        print(f"\n✓ All assertions passed")
        return True
    
    except Exception as e:
        print(f"\n✗ Error testing batch collation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_windows(data_root: str):
    """
    Test loading multiple window types.
    
    Args:
        data_root: Root directory containing processed CT data
    """
    print("\n" + "=" * 80)
    print("Test 4: Multiple Window Types")
    print("=" * 80)
    
    window_types = ['lung', 'bone', 'soft_tissue', 'brain']
    results = {}
    
    for window_type in window_types:
        try:
            dataset = CTWindowSparseSDF(
                roots=data_root,
                resolution=512,
                window_type=window_type,
                min_points=100,
                max_points=500000
            )
            
            results[window_type] = {
                'success': True,
                'num_instances': len(dataset)
            }
            
            if len(dataset) > 0:
                sample = dataset[0]
                results[window_type]['num_points'] = sample['sparse_sdf'].shape[0]
            
        except Exception as e:
            results[window_type] = {
                'success': False,
                'error': str(e)
            }
    
    print("\nResults:")
    for window_type, result in results.items():
        if result['success']:
            status = "✓"
            info = f"{result['num_instances']} instances"
            if 'num_points' in result:
                info += f", ~{result['num_points']} points/sample"
        else:
            status = "✗"
            info = f"Error: {result['error']}"
        
        print(f"  {status} {window_type:15s}: {info}")
    
    return all(r['success'] for r in results.values())


def test_dataloader_iteration(dataset: CTWindowSparseSDF, num_batches: int = 3):
    """
    Test iterating through multiple batches.
    
    Args:
        dataset: CTWindowSparseSDF instance
        num_batches: Number of batches to iterate
    """
    print("\n" + "=" * 80)
    print(f"Test 5: DataLoader Iteration ({num_batches} batches)")
    print("=" * 80)
    
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_fn
        )
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            print(f"\n  Batch {i+1}:")
            print(f"    Total points: {batch['sparse_sdf'].shape[0]}")
            print(f"    Batch indices: {torch.unique(batch['batch_idx']).tolist()}")
        
        print(f"\n✓ Successfully iterated through {num_batches} batches")
        return True
    
    except Exception as e:
        print(f"\n✗ Error during iteration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CT Window Sparse Dataset')
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing processed CT data'
    )
    parser.add_argument(
        '--window_type',
        type=str,
        default='lung',
        choices=['lung', 'bone', 'soft_tissue', 'brain'],
        help='Window type to test'
    )
    parser.add_argument(
        '--all_windows',
        action='store_true',
        help='Test all window types'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("CT Window Sparse Dataset Test Suite")
    print("=" * 80)
    print(f"\nData root: {args.data_root}")
    print(f"Window type: {args.window_type}")
    
    # Test 1: Basic loading
    success = test_basic_loading(args.data_root, args.window_type)
    if not success:
        print("\n✗ Basic loading test failed. Aborting.")
        return
    
    # Create dataset for subsequent tests
    dataset = CTWindowSparseSDF(
        roots=args.data_root,
        resolution=512,
        window_type=args.window_type,
        min_points=100,
        max_points=500000
    )
    
    # Test 2: Single sample
    test_single_sample(dataset)
    
    # Test 3: Batch collation
    test_batch_collation(dataset)
    
    # Test 4: Multiple windows (optional)
    if args.all_windows:
        test_multiple_windows(args.data_root)
    
    # Test 5: DataLoader iteration
    test_dataloader_iteration(dataset)
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

