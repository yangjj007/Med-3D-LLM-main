"""
Project 3D latent [N, 16] to LLM hidden size [N, hidden_size].
Optional 3D positional encoding from coords (x, y, z).
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class PositionEncoder3D(nn.Module):
    """
    Optional 3D sinusoidal or learned positional encoding from coords [N, 4].
    Uses (x, y, z) from coords; batch_idx is ignored for position.
    """

    def __init__(
        self,
        hidden_size: int,
        max_coord: int = 64,
        mode: str = "sinusoidal",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_coord = max_coord
        self.mode = mode
        if mode == "learned":
            self.embed = nn.Embedding((max_coord + 1) ** 3, hidden_size)
        else:
            self.embed = None

    def _sinusoidal_3d(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N, 4] (batch, x, y, z); use x,y,z in [0, max_coord]
        xyz = coords[:, 1:4].float()
        N = coords.shape[0]
        d = self.hidden_size // 3
        d = (d // 2) * 2
        dims = [d, d, self.hidden_size - 2 * d]
        pe = []
        for i in range(3):
            di = (dims[i] // 2) * 2
            if di <= 0:
                continue
            div = torch.exp(torch.arange(0, di, 2, device=coords.device).float() * (-math.log(10000.0) / di))
            pos = xyz[:, i : i + 1]
            pe_i = torch.zeros(N, di, device=coords.device, dtype=coords.dtype)
            pe_i[:, 0::2] = torch.sin(pos * div)
            pe_i[:, 1::2] = torch.cos(pos * div)
            pe.append(pe_i)
        out = torch.cat(pe, dim=-1)
        if out.shape[-1] < self.hidden_size:
            out = torch.cat([out, torch.zeros(N, self.hidden_size - out.shape[-1], device=coords.device, dtype=out.dtype)], dim=-1)
        return out

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [N, 4] or [B, N, 4], (batch_idx, x, y, z).
        Returns:
            [N, hidden_size] or [B, N, hidden_size] position encoding.
        """
        if coords.dim() == 3:
            B, N, _ = coords.shape
            coords_flat = coords.reshape(B * N, 4)
            out = self._forward_flat(coords_flat)
            return out.reshape(B, N, self.hidden_size)
        return self._forward_flat(coords)

    def _forward_flat(self, coords: torch.Tensor) -> torch.Tensor:
        if self.mode == "sinusoidal":
            return self._sinusoidal_3d(coords)
        else:
            x, y, z = coords[:, 1].long().clamp(0, self.max_coord), \
                      coords[:, 2].long().clamp(0, self.max_coord), \
                      coords[:, 3].long().clamp(0, self.max_coord)
            idx = x * (self.max_coord + 1) ** 2 + y * (self.max_coord + 1) + z
            return self.embed(idx)


class Projector3D(nn.Module):
    """
    Project 3D latent feats [..., 16] to LLM hidden size [..., hidden_size].
    Optional: add 3D position encoding from coords.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_size: int = 1024,
        num_layers: int = 1,
        use_3d_pos: bool = False,
        pos_mode: str = "sinusoidal",
        max_coord: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        if num_layers == 1:
            self.proj = nn.Linear(latent_dim, hidden_size)
        else:
            layers = [nn.Linear(latent_dim, hidden_size), nn.GELU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
            layers.append(nn.Linear(hidden_size, hidden_size))
            self.proj = nn.Sequential(*layers)
        self.use_3d_pos = use_3d_pos
        self.pos_encoder = PositionEncoder3D(hidden_size, max_coord, pos_mode) if use_3d_pos else None

    def forward(
        self,
        feats: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feats: [N, 16] or [B, N, 16].
            coords: [N, 4] or [B, N, 4] optional; used only if use_3d_pos.
        Returns:
            [N, hidden_size] or [B, N, hidden_size].
        """
        out = self.proj(feats)
        if self.use_3d_pos and self.pos_encoder is not None and coords is not None:
            out = out + self.pos_encoder(coords)
        return out
