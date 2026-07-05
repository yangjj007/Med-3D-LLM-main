import os
import torch
import torch.nn as nn
from .. import SparseTensor
from torchsparse.utils import make_ntuple

# 调试 ImplicitGEMM：export DECODER_TORCHSPARSE_DEBUG=1 或 DECODER_SPARSE_CONV_DEBUG=1


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True", "yes", "YES")


def _sparse_conv_ts_debug_enabled() -> bool:
    """torchsparse 卷积前后细日志：DECODER_TORCHSPARSE_DEBUG 或 DECODER_SPARSE_CONV_DEBUG=1。"""
    return _env_true("DECODER_TORCHSPARSE_DEBUG") or _env_true("DECODER_SPARSE_CONV_DEBUG")


def _debug_dump_before_torchsparse_conv(tag: str, mod: nn.Module, x: SparseTensor) -> None:
    if not _sparse_conv_ts_debug_enabled():
        return
    d = x.data
    C, Fm = d.C, d.F
    n = int(C.shape[0])
    lines = [
        f"[TSDBG conv] {tag}",
        f"  module: in={getattr(mod, 'conv', mod).in_channels if hasattr(mod, 'conv') else '?'} "
        f"out={getattr(mod, 'conv', mod).out_channels if hasattr(mod, 'conv') else '?'} "
        f"kernel={getattr(mod, 'conv', mod).kernel_size if hasattr(mod, 'conv') else '?'} "
        f"stride={getattr(mod, 'conv', mod).stride if hasattr(mod, 'conv') else '?'} "
        f"padding={getattr(mod, 'conv', mod).padding if hasattr(mod, 'conv') else '?'} "
        f"transposed={getattr(mod, 'conv', mod).transposed if hasattr(mod, 'conv') else '?'}",
        f"  training={mod.training}  N={n}  F.shape={tuple(Fm.shape)} F.dtype={Fm.dtype} C.dtype={C.dtype}",
    ]
    if n > 0 and C.ndim == 2 and C.shape[1] >= 4:
        xyz = C[:, 1:4].long()
        lines.append(
            f"  xyz min=[{xyz[:,0].min().item()},{xyz[:,1].min().item()},{xyz[:,2].min().item()}] "
            f"max=[{xyz[:,0].max().item()},{xyz[:,1].max().item()},{xyz[:,2].max().item()}]"
        )
        sample_n = int(os.environ.get("DECODER_TSDBG_COORD_SAMPLE", "500000"))
        sample_n = max(1000, min(sample_n, n))
        sub = C[:sample_n]
        uniq = torch.unique(sub, dim=0).shape[0]
        dup = sample_n - int(uniq)
        lines.append(
            f"  coord_dup_estimate(first_{sample_n}): unique={int(uniq)} dup≈{dup} ratio≈{dup / sample_n:.6f}"
        )
    for attr in ("spatial_range", "stride"):
        if hasattr(d, attr):
            try:
                lines.append(f"  ts.{attr}={getattr(d, attr)}")
            except Exception as e:  # noqa: BLE001
                lines.append(f"  ts.{attr}=<err {e}>")
    w = getattr(mod, "conv", None)
    if w is not None and hasattr(w, "weight"):
        wt = w.weight
        lines.append(
            f"  weight dtype={wt.dtype} shape={tuple(wt.shape)} "
            f"finite={torch.isfinite(wt).all().item()}"
        )
    print("\n".join(lines), flush=True)


def sparseconv3d_func(input: SparseTensor, weight: torch.Tensor, kernel_size: int, stride: int = 1, dilation: int = 1, padding: int = 0, bias: torch.Tensor = None, training: bool = True):
    if 'torchsparse' not in globals():
        import torchsparse
    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    _padding = make_ntuple(padding, 3)
    padding = ()
    for i in range(3):
        if kernel_size[i] % 2 == 1 and stride[i] == 1:
            padding += ((kernel_size[i] - 1) // 2,)
        else:
            padding += (_padding[i],)
    out = torchsparse.nn.functional.conv3d(input.data, weight, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, training=training)
    spatial_range = out.spatial_range
    new_shape = [input.shape[0], weight.shape[1]]
    out = SparseTensor(out, shape=torch.Size(new_shape), layout=input.layout if all(s == 1 for s in stride) else None)
    out._spatial_cache = input._spatial_cache
    out._scale = tuple([s * stride for s, stride in zip(input._scale, stride)])
    out.data.spatial_range = spatial_range
    return out

class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if 'torchsparse' not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

    def forward(self, x: SparseTensor) -> SparseTensor:
        _debug_dump_before_torchsparse_conv(
            f"SparseConv3d.forward id={id(self)}", self, x
        )
        out = self.conv(x.data)

        spatial_range = out.spatial_range

        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
        out._spatial_cache = x._spatial_cache
        out._scale = tuple([s * stride for s, stride in zip(x._scale, self.conv.stride)])

        out.data.spatial_range = spatial_range

        if _sparse_conv_ts_debug_enabled():
            d = out.data
            print(
                f"[TSDBG conv] SparseConv3d.forward id={id(self)} | "
                f"OUT N={d.C.shape[0]} F.shape={tuple(d.F.shape)} spatial_range={getattr(d, 'spatial_range', '?')}",
                flush=True,
            )

        return out


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if 'torchsparse' not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias, transposed=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        _debug_dump_before_torchsparse_conv(
            f"SparseInverseConv3d.forward id={id(self)}", self, x
        )
        out = self.conv(x.data)        

        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
        out._spatial_cache = x._spatial_cache
        out._scale = tuple([s // stride for s, stride in zip(x._scale, self.conv.stride)])
        
        return out



