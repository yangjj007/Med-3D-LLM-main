from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from ...modules.utils import convert_module_to_f16, convert_module_to_f32
from .base import SparseTransformerBase


class SparseDownBlock3d(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )

        self.down = sp.SparseDownsample(2)
        # Use padding=None to get SubMConv3d (preserves voxel grid for spconv backend skip connection)
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            sp.SparseConv3d(self.out_channels, self.out_channels, 3),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1)

        self.use_checkpoint = use_checkpoint
        
    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.down(h)
        x = self.down(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseSDFEncoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )

        self.input_layer1 = sp.SparseLinear(1, model_channels // 16)
        
        self.downsample = nn.ModuleList([
            SparseDownBlock3d(
                channels=model_channels//16,
                out_channels=model_channels // 8,
                use_checkpoint=use_checkpoint,
            ),
            SparseDownBlock3d(
                channels=model_channels // 8,
                out_channels=model_channels // 4,
                use_checkpoint=use_checkpoint,
            ),
            SparseDownBlock3d(
                channels=model_channels // 4,
                out_channels=model_channels,
                use_checkpoint=use_checkpoint,
            )
        ])

        self.resolution = resolution
        # VQVAE: 输出2*latent_channels保持与VAE架构兼容，但forward只返回mean
        self.out_layer = sp.SparseLinear(model_channels, latent_channels * 2)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # # Zero-out output layers:
        # nn.init.constant_(self.out_layer.weight, 0)
        # nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        super().convert_to_fp16()
        self.input_layer1.apply(convert_module_to_f16)
        self.downsample.apply(convert_module_to_f16)
        self.out_layer.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        super().convert_to_fp32()
        self.input_layer1.apply(convert_module_to_f32)
        self.downsample.apply(convert_module_to_f32)
        self.out_layer.apply(convert_module_to_f32)

    def forward(self, x: sp.SparseTensor, factor: float = None):
        print(f"[DEBUG Encoder.forward] Input x.shape: {x.shape}, x.feats.shape: {x.feats.shape}")
        print(f"[DEBUG Encoder.forward] Input x.feats min: {x.feats.min().item():.6f}, max: {x.feats.max().item():.6f}, mean: {x.feats.mean().item():.6f}")
        
        # Convert input to the correct dtype for fp16 mode
        dtype = x.feats.dtype
        x = x.type(self.dtype)
        
        x = self.input_layer1(x)
        print(f"[DEBUG Encoder.forward] After input_layer1, x.feats.shape: {x.feats.shape}")
        print(f"[DEBUG Encoder.forward] After input_layer1, x.feats min: {x.feats.min().item():.6f}, max: {x.feats.max().item():.6f}, mean: {x.feats.mean().item():.6f}")
        
        for idx, block in enumerate(self.downsample):
            x = block(x)
            print(f"[DEBUG Encoder.forward] After downsample block {idx}, x.feats.shape: {x.feats.shape}")
            print(f"[DEBUG Encoder.forward] After downsample block {idx}, x.feats min: {x.feats.min().item():.6f}, max: {x.feats.max().item():.6f}, mean: {x.feats.mean().item():.6f}")
        
        print(f"[DEBUG Encoder.forward] Before super().forward, x.feats.shape: {x.feats.shape}")
        h = super().forward(x, factor)
        print(f"[DEBUG Encoder.forward] After super().forward, h.feats.shape: {h.feats.shape}")
        print(f"[DEBUG Encoder.forward] After super().forward, h.feats min: {h.feats.min().item():.6f}, max: {h.feats.max().item():.6f}, mean: {h.feats.mean().item():.6f}")
        
        h = h.type(dtype)
        print(f"[DEBUG Encoder.forward] After type conversion, h.feats min: {h.feats.min().item():.6f}, max: {h.feats.max().item():.6f}")
        
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        print(f"[DEBUG Encoder.forward] After layer_norm, h.feats min: {h.feats.min().item():.6f}, max: {h.feats.max().item():.6f}, mean: {h.feats.mean().item():.6f}")
        
        h = self.out_layer(h)
        print(f"[DEBUG Encoder.forward] After out_layer, h.feats.shape: {h.feats.shape}")
        print(f"[DEBUG Encoder.forward] After out_layer, h.feats min: {h.feats.min().item():.6f}, max: {h.feats.max().item():.6f}, mean: {h.feats.mean().item():.6f}")
        
        # VQVAE: 分割成mean和logvar，但只返回mean（ShapeLLM方法）
        mean_feats, logvar_feats = torch.chunk(h.feats, 2, dim=-1)
        print(f"[DEBUG Encoder.forward] After chunk, mean.shape: {mean_feats.shape}, logvar.shape: {logvar_feats.shape}")
        print(f"[DEBUG Encoder.forward] Returning only mean (logvar discarded)")
        
        h_mean = h.replace(mean_feats)
        return h_mean