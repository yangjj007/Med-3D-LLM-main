"""
Trainer for Sparse SDF VQVAE model.
"""

from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...modules import sparse as sp


class SparseSDF_VQVAETrainer(BasicTrainer):
    """
    Trainer for Sparse SDF VQVAE.
    
    Args:
        models (dict[str, nn.Module]): Models to train. Should contain 'vqvae' key.
        dataset (torch.utils.data.Dataset): Dataset returning sparse SDF data.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        lambda_vq (float): VQ loss weight.
        lambda_commitment (float): Commitment loss weight.
        loss_type (str): Reconstruction loss type. 'mse' for MSE loss, 'l1' for L1 loss, 'l1_l2' for combined L1+L2 loss.
        pretrained_vae_path (str): Path to pretrained VAE checkpoint to initialize from.
        training_stage (int): Training stage. 1 = freeze VAE, train codebook only; 2 = joint training (default: 1).
    """
    
    def __init__(
        self,
        *args,
        lambda_vq: float = 1.0,
        lambda_commitment: float = 0.25,
        loss_type: str = 'mse',
        pretrained_vae_path: str = None,
        training_stage: int = 1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_vq = lambda_vq
        self.lambda_commitment = lambda_commitment
        self.loss_type = loss_type
        self.training_stage = training_stage
        
        # Load pretrained VAE if specified
        if pretrained_vae_path is not None and self.step == 0:
            self._load_pretrained_vae(pretrained_vae_path)
        
        # Apply training stage configuration
        self._configure_training_stage()
    
    def _load_pretrained_vae(self, pretrained_vae_path: str):
        """
        Load pretrained VAE weights.
        
        Args:
            pretrained_vae_path: Path to pretrained VAE checkpoint (.pth file)
        """
        if self.is_master:
            print(f'\nLoading pretrained VAE from: {pretrained_vae_path}')
        
        # Load checkpoint
        checkpoint = torch.load(pretrained_vae_path, map_location='cpu', weights_only=True)
        
        # Extract VAE state dict
        if 'vae' in checkpoint:
            vae_state_dict = checkpoint['vae']
        elif 'state_dict' in checkpoint:
            vae_state_dict = checkpoint['state_dict']
        else:
            vae_state_dict = checkpoint
        
        # Get encoder and decoder state dicts
        encoder_state_dict = {
            k.replace('encoder.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('encoder.')
        }
        decoder_state_dict = {
            k.replace('decoder.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('decoder.')
        }
        
        # Load into VQVAE model
        vqvae = self.models['vqvae']
        if hasattr(vqvae, 'module'):
            vqvae = vqvae.module
        
        if hasattr(vqvae, 'load_pretrained_vae'):
            vqvae.load_pretrained_vae(encoder_state_dict, decoder_state_dict)
            if self.is_master:
                print('Successfully loaded pretrained VAE weights')
        else:
            if self.is_master:
                print('Warning: VQVAE model does not have load_pretrained_vae method')
    
    def _configure_training_stage(self):
        """
        Configure model parameters based on training stage.
        
        Stage 1: Freeze encoder and decoder, train only codebook
        Stage 2: Unfreeze all parameters for joint training
        """
        vqvae = self.models['vqvae']
        if hasattr(vqvae, 'module'):
            vqvae = vqvae.module
        
        if self.training_stage == 1:
            # Stage 1: Freeze encoder and decoder
            if hasattr(vqvae, 'encoder'):
                for param in vqvae.encoder.parameters():
                    param.requires_grad = False
            
            if hasattr(vqvae, 'decoder'):
                for param in vqvae.decoder.parameters():
                    param.requires_grad = False
            
            # Ensure VQ parameters are trainable
            if hasattr(vqvae, 'vq'):
                for param in vqvae.vq.parameters():
                    param.requires_grad = True
            
            if self.is_master:
                print("\n" + "=" * 80)
                print("[Stage 1] Encoder and Decoder frozen, training Codebook only")
                print("=" * 80)
                
                # Count trainable parameters
                total_params = sum(p.numel() for p in vqvae.parameters())
                trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"Frozen parameters: {total_params - trainable_params:,}")
                print("=" * 80 + "\n")
        
        elif self.training_stage == 2:
            # Stage 2: Unfreeze all parameters
            for param in vqvae.parameters():
                param.requires_grad = True
            
            if self.is_master:
                print("\n" + "=" * 80)
                print("[Stage 2] Joint training: Encoder + Decoder + Codebook")
                print("=" * 80)
                
                # Count trainable parameters
                total_params = sum(p.numel() for p in vqvae.parameters())
                trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print("=" * 80 + "\n")
        
        else:
            raise ValueError(f"Invalid training_stage: {self.training_stage}. Must be 1 or 2.")
    
    def training_losses(
        self,
        sparse_sdf: torch.Tensor,
        sparse_index: torch.Tensor,
        batch_idx: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.
        
        Args:
            sparse_sdf: SDF values [total_N, 1]
            sparse_index: 3D coordinates [total_N, 3]
            batch_idx: Batch indices [total_N]
        
        Returns:
            Tuple of (loss_dict, status_dict)
        """
        # Construct sparse tensor input
        coords = torch.cat([batch_idx.unsqueeze(-1), sparse_index], dim=-1).int()
        x = sp.SparseTensor(sparse_sdf, coords)
        
        # Forward pass through VQVAE
        vqvae = self.training_models['vqvae']
        recon, vq_loss, commitment_loss = vqvae(x)
        
        # Compute reconstruction loss
        if self.loss_type == 'mse':
            recon_loss = F.mse_loss(recon.feats, sparse_sdf, reduction='mean')
        elif self.loss_type == 'l1':
            recon_loss = F.l1_loss(recon.feats, sparse_sdf, reduction='mean')
        elif self.loss_type == 'l1_l2':
            # Combined L1 + L2 loss for stage 2
            recon_loss = 0.5 * F.l1_loss(recon.feats, sparse_sdf, reduction='mean') + \
                         0.5 * F.mse_loss(recon.feats, sparse_sdf, reduction='mean')
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')
        
        # Total loss
        total_loss = recon_loss + self.lambda_vq * vq_loss + self.lambda_commitment * commitment_loss
        
        # Loss dictionary
        terms = edict(
            loss=total_loss,
            recon=recon_loss,
            vq=vq_loss,
            commitment=commitment_loss,
        )
        
        return terms, {}
    
    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=16, batch_size=None, verbose=False):
        """Take a snapshot of the model's performance."""
        # Use training batch_size if not specified, default to 2 for safety
        if batch_size is None:
            batch_size = getattr(self, 'batch_size_per_gpu', 2)
        print(f"[DEBUG] snapshot: Using batch_size={batch_size}, num_samples={num_samples}")
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Run snapshot inference.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for inference
            verbose: Whether to print verbose output
        
        Returns:
            Dictionary of samples for visualization
        """
        try:
            print(f"\n[DEBUG] run_snapshot called with num_samples={num_samples}, batch_size={batch_size}")
            
            # Create a dataset copy with potentially reduced max_points for stability
            dataset_copy = copy.deepcopy(self.dataset)
            original_max_points = getattr(dataset_copy, 'max_points', None)
            
            
            dataloader = DataLoader(
                dataset_copy,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=dataset_copy.collate_fn if hasattr(dataset_copy, 'collate_fn') else None,
            )
            print(f"[DEBUG] DataLoader created with max_points={getattr(dataset_copy, 'max_points', 'N/A')}")
            
            # Get VQVAE model
            vqvae = self.models['vqvae']
            if hasattr(vqvae, 'module'):
                vqvae = vqvae.module
            
            # Get model dtype for fp16 compatibility
            model_dtype = vqvae.encoder.dtype if hasattr(vqvae.encoder, 'dtype') else torch.float32
            print(f"[DEBUG] Model dtype: {model_dtype}")
            
            # Inference
            gts = []
            recons = []
            
            for i in range(0, num_samples, batch_size):
                batch = min(batch_size, num_samples - i)
                print(f"[DEBUG] Processing iteration {i}, batch={batch}")
                
                data = next(iter(dataloader))
                print(f"[DEBUG] Data loaded from dataloader")
                print(f"[DEBUG] Data keys: {data.keys()}")
                
                # Move to device (data already collated, no need to slice)
                sparse_sdf = data['sparse_sdf'].cuda() if isinstance(data['sparse_sdf'], torch.Tensor) else data['sparse_sdf']
                sparse_index = data['sparse_index'].cuda() if isinstance(data['sparse_index'], torch.Tensor) else data['sparse_index']
                batch_idx = data['batch_idx'].cuda() if isinstance(data['batch_idx'], torch.Tensor) else data['batch_idx']
                
                print(f"[DEBUG] After moving to device:")
                print(f"[DEBUG]   sparse_sdf shape: {sparse_sdf.shape}")
                print(f"[DEBUG]   sparse_index shape: {sparse_index.shape}")
                print(f"[DEBUG]   batch_idx shape: {batch_idx.shape}")
                
                # Only keep points belonging to current batch
                mask = batch_idx < batch
                print(f"[DEBUG] Mask sum: {mask.sum().item()} / {len(mask)}")
                
                sparse_sdf = sparse_sdf[mask]
                sparse_index = sparse_index[mask]
                batch_idx = batch_idx[mask]
                
                print(f"[DEBUG] After masking:")
                print(f"[DEBUG]   sparse_sdf shape: {sparse_sdf.shape}")
                print(f"[DEBUG]   sparse_index shape: {sparse_index.shape}")
                print(f"[DEBUG]   batch_idx shape: {batch_idx.shape}")
                
                # Convert to model dtype (for fp16 compatibility)
                sparse_sdf = sparse_sdf.to(dtype=model_dtype)
                
                # Construct sparse tensor
                coords = torch.cat([batch_idx.unsqueeze(-1), sparse_index], dim=-1).int()
                print(f"[DEBUG] coords shape: {coords.shape}, dtype: {coords.dtype}")
                print(f"[DEBUG] coords min: {coords.min(dim=0).values}, max: {coords.max(dim=0).values}")
                print(f"[DEBUG] coords is_integer: {coords.dtype in [torch.int32, torch.int64]}")
                print(f"[DEBUG] sparse_sdf shape: {sparse_sdf.shape}, dtype: {sparse_sdf.dtype}")
                print(f"[DEBUG] sparse_sdf min: {sparse_sdf.min()}, max: {sparse_sdf.max()}, mean: {sparse_sdf.mean()}")
                
                x = sp.SparseTensor(sparse_sdf, coords)
                print(f"[DEBUG] SparseTensor created")
                print(f"[DEBUG] SparseTensor shape: {x.shape}")
                print(f"[DEBUG] SparseTensor feats shape: {x.feats.shape}")
                print(f"[DEBUG] SparseTensor coords shape: {x.coords.shape}")
                
                # Check for any NaN or Inf
                if torch.isnan(sparse_sdf).any():
                    print(f"[ERROR] NaN detected in sparse_sdf!")
                if torch.isinf(sparse_sdf).any():
                    print(f"[ERROR] Inf detected in sparse_sdf!")
                
                # Encode and decode
                print(f"[DEBUG] Calling vqvae.Encode...")
                try:
                    # 构建字典格式的 batch，符合 Encode 方法的输入要求
                    batch_dict = {
                        'sparse_sdf': sparse_sdf,
                        'sparse_index': sparse_index,
                        'batch_idx': batch_idx,
                    }
                    
                    # ===== DEBUG: 详细检查 batch_dict =====
                    print(f"\n[DEBUG run_snapshot] Before calling Encode:")
                    print(f"[DEBUG run_snapshot] batch_dict type: {type(batch_dict)}")
                    print(f"[DEBUG run_snapshot] batch_dict is dict: {isinstance(batch_dict, dict)}")
                    print(f"[DEBUG run_snapshot] batch_dict keys: {batch_dict.keys()}")
                    print(f"[DEBUG run_snapshot] batch_dict['sparse_sdf'] type: {type(batch_dict['sparse_sdf'])}")
                    print(f"[DEBUG run_snapshot] batch_dict['sparse_sdf'] shape: {batch_dict['sparse_sdf'].shape}")
                    print(f"[DEBUG run_snapshot] batch_dict['sparse_index'] type: {type(batch_dict['sparse_index'])}")
                    print(f"[DEBUG run_snapshot] batch_dict['sparse_index'] shape: {batch_dict['sparse_index'].shape}")
                    print(f"[DEBUG run_snapshot] batch_dict['batch_idx'] type: {type(batch_dict['batch_idx'])}")
                    print(f"[DEBUG run_snapshot] batch_dict['batch_idx'] shape: {batch_dict['batch_idx'].shape}")
                    print("=" * 80)
                    # ===== DEBUG END =====
                    
                    encoding_indices = vqvae.Encode(batch_dict)
                    print(f"[DEBUG] Encoding done")
                except FloatingPointError as e:
                    print(f"[ERROR] FloatingPointError in Encode: {e}")
                    raise
                except RuntimeError as e:
                    print(f"[ERROR] RuntimeError in Encode: {e}")
                    raise
                except Exception as e:
                    print(f"[ERROR] Unexpected error in Encode: {type(e).__name__}: {e}")
                    raise
                
                recon = vqvae.Decode(encoding_indices)
                print(f"[DEBUG] Decoding done")
                
                # Store results
                gts.append({
                    'sparse_sdf': sparse_sdf,
                    'sparse_index': sparse_index,
                    'batch_idx': batch_idx,
                })
                recons.append({
                    'sparse_sdf': recon.feats,
                    'sparse_index': recon.coords[:, 1:],
                    'batch_idx': recon.coords[:, 0],
                })
                print(f"[DEBUG] Results stored for iteration {i}")
            
            # Combine results
            print(f"[DEBUG] Combining {len(gts)} ground truth results")
            gt_combined = {
                'sparse_sdf': torch.cat([g['sparse_sdf'] for g in gts], dim=0),
                'sparse_index': torch.cat([g['sparse_index'] for g in gts], dim=0),
                'batch_idx': torch.cat([g['batch_idx'] for g in gts], dim=0),
            }
            print(f"[DEBUG] GT combined shapes: sdf={gt_combined['sparse_sdf'].shape}, "
                  f"index={gt_combined['sparse_index'].shape}, batch={gt_combined['batch_idx'].shape}")
            
            print(f"[DEBUG] Combining {len(recons)} reconstruction results")
            recon_combined = {
                'sparse_sdf': torch.cat([r['sparse_sdf'] for r in recons], dim=0),
                'sparse_index': torch.cat([r['sparse_index'] for r in recons], dim=0),
                'batch_idx': torch.cat([r['batch_idx'] for r in recons], dim=0),
            }
            print(f"[DEBUG] Recon combined shapes: sdf={recon_combined['sparse_sdf'].shape}, "
                  f"index={recon_combined['sparse_index'].shape}, batch={recon_combined['batch_idx'].shape}")
            
            sample_dict = {
                'gt': {'value': gt_combined, 'type': 'sample'},
                'recon': {'value': recon_combined, 'type': 'sample'},
            }
            
            print(f"[DEBUG] run_snapshot completed successfully")
            return sample_dict
            
        except Exception as e:
            print(f"\n[ERROR] Exception in run_snapshot:")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            print(f"[ERROR] Exception message: {str(e)}")
            import traceback
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            raise

