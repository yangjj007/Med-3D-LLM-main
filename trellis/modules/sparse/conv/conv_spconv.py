import torch
import torch.nn as nn
from .. import SparseTensor
from .. import DEBUG
from . import SPCONV_ALGO

class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        algo = None
        if SPCONV_ALGO == 'native':
            algo = spconv.ConvAlgo.Native
        elif SPCONV_ALGO == 'implicit_gemm':
            algo = spconv.ConvAlgo.MaskImplicitGemm
        if stride == 1 and (padding is None):
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
        else:
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self.padding = padding

    def forward(self, x: SparseTensor) -> SparseTensor:
        # [DEBUG] 检查输入数据类型
        print(f"[DEBUG SparseConv3d.forward] Input coords dtype: {x.coords.dtype}, shape: {x.coords.shape}")
        print(f"[DEBUG SparseConv3d.forward] Input feats dtype: {x.feats.dtype}, shape: {x.feats.shape}")
        print(f"[DEBUG SparseConv3d.forward] Input coords min: {x.coords.min(dim=0).values}, max: {x.coords.max(dim=0).values}")
        
        if not torch.isfinite(x.feats).all():
            # 打印非法值位置和统计
            nan_mask = torch.isnan(x.feats)
            inf_mask = torch.isinf(x.feats)
            print(f"[CRITICAL] NaN count: {nan_mask.sum().item()}, Inf count: {inf_mask.sum().item()}")
            print(f"[CRITICAL] Features stats - mean: {x.feats[torch.isfinite(x.feats)].mean():.4f}, "
                f"std: {x.feats[torch.isfinite(x.feats)].std():.4f}")
        
        feat_abs_max = x.feats.abs().max()
        if feat_abs_max > 1e4:
            print(f"[WARNING] Large feature values! Max: {feat_abs_max:.2e}.")
        
        # 检查是否有 NaN 或 Inf
        if torch.isnan(x.feats).any():
            print(f"[ERROR] NaN detected in feats!")
        if torch.isinf(x.feats).any():
            print(f"[ERROR] Inf detected in feats!")
        
        spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
        print(f"[DEBUG SparseConv3d.forward] Calling spconv with stride={self.stride}, padding={self.padding}")
        new_data = self.conv(x.data)
        print(f"[DEBUG SparseConv3d.forward] spconv call succeeded")
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout

        if spatial_changed and (x.shape[0] != 1):
            # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
            fwd = new_data.indices[:, 0].argsort()
            bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
            sorted_feats = new_data.features[fwd]
            sorted_coords = new_data.indices[fwd]
            unsorted_data = new_data
            new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore

        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )

        if spatial_changed and (x.shape[0] != 1):
            out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
            out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)
 
        return out


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride)
        if spatial_changed:
            # recover the original spconv order
            data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
            bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
            data = data.replace_feature(x.feats[bwd])
            if DEBUG:
                assert torch.equal(data.indices, x.coords[bwd]), 'Recover the original order failed'
        else:
            data = x.data

        new_data = self.conv(data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout
        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )
        return out
