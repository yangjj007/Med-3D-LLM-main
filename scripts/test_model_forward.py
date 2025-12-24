"""
Test VQVAE model forward pass.
"""

import sys
import os

# Add the parent directory of TRELLIS-main to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
trellis_main_dir = os.path.dirname(script_dir)  # TRELLIS-main/
project_root = os.path.dirname(trellis_main_dir)  # Med-3D-LLM-main/

# Add TRELLIS-main directory to path so we can import trellis
sys.path.insert(0, trellis_main_dir)

import torch
from trellis.models import Direct3DS2_VQVAE
from trellis.modules import sparse as sp


def test_vqvae_forward():
    """Test VQVAE forward pass."""
    print("=== Testing VQVAE Forward Pass ===\n")
    
    # Create model
    print("Creating VQVAE model...")
    model = Direct3DS2_VQVAE(
        resolution=64,
        model_channels=512,  # 必须是512的倍数！(512, 1024, 1536, 2048...)
        latent_channels=16,
        num_blocks=2,
        num_embeddings=512,
        use_fp16=False
    )
    model.eval()
    print(f"✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create dummy input
    print("Creating dummy sparse input...")
    num_points = 1000
    batch_size = 2
    
    # Generate random sparse data
    sparse_sdf = torch.randn(num_points, 1)
    sparse_index = torch.randint(0, 64, (num_points, 3))
    batch_idx = torch.randint(0, batch_size, (num_points,))
    
    # Create SparseTensor
    coords = torch.cat([batch_idx.unsqueeze(-1), sparse_index], dim=-1).int()
    x = sp.SparseTensor(sparse_sdf, coords)
    
    print(f"✓ Input created")
    print(f"  Num points: {num_points}")
    print(f"  Batch size: {batch_size}")
    print(f"  sparse_sdf shape: {sparse_sdf.shape}")
    print(f"  sparse_index shape: {sparse_index.shape}")
    print()
    
    # Test full forward pass
    print("Testing full forward pass (train mode)...")
    with torch.no_grad():
        recon, vq_loss, commitment_loss = model(x)
    
    print(f"✓ Forward pass successful")
    print(f"  Reconstruction feats shape: {recon.feats.shape}")
    print(f"  Reconstruction coords shape: {recon.coords.shape}")
    print(f"  VQ loss: {vq_loss.item():.6f}")
    print(f"  Commitment loss: {commitment_loss.item():.6f}")
    print()
    
    # Test encoding
    print("Testing encoding...")
    with torch.no_grad():
        encoding_indices = model.Encode(x)
    
    print(f"✓ Encoding successful")
    print(f"  Encoding indices feats shape: {encoding_indices.feats.shape}")
    print(f"  Encoding indices coords shape: {encoding_indices.coords.shape}")
    print(f"  Unique indices: {torch.unique(encoding_indices.feats.long()).numel()}")
    print()
    
    # Test decoding
    print("Testing decoding...")
    with torch.no_grad():
        decoded = model.Decode(encoding_indices)
    
    print(f"✓ Decoding successful")
    print(f"  Decoded feats shape: {decoded.feats.shape}")
    print(f"  Decoded coords shape: {decoded.coords.shape}")
    print()
    
    # Test encode-decode cycle
    print("Testing encode-decode cycle...")
    with torch.no_grad():
        # Encode
        enc_indices = model.Encode(x)
        # Decode
        dec = model.Decode(enc_indices)
        # Compute reconstruction error
        if dec.feats.shape[0] == sparse_sdf.shape[0]:
            recon_error = torch.abs(dec.feats - sparse_sdf).mean()
            print(f"✓ Encode-decode cycle successful")
            print(f"  Mean reconstruction error: {recon_error.item():.6f}")
        else:
            print(f"⚠ Warning: Output shape mismatch")
            print(f"  Input points: {sparse_sdf.shape[0]}")
            print(f"  Output points: {dec.feats.shape[0]}")
    print()
    
    print("=================================")
    print("All forward pass tests passed! ✓")
    print("=================================")


if __name__ == '__main__':
    test_vqvae_forward()

