import torch
import torch.nn as nn
from . import SparseTensor

__all__ = [
    'SparseLinear'
]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        print(f"[DEBUG SparseLinear] 输入 feats.shape: {input.feats.shape}")
        print(f"[DEBUG SparseLinear] 权重 weight.shape: {self.weight.shape}")
        print(f"[DEBUG SparseLinear] 期望输入维度: {self.in_features}, 实际输入维度: {input.feats.shape[-1]}")
        if input.feats.shape[-1] != self.in_features:
            print(f"[DEBUG SparseLinear] ❌ 维度不匹配！无法进行矩阵乘法！")
        return input.replace(super().forward(input.feats))
