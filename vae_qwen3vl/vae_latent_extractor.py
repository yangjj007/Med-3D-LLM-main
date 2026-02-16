"""
Extract 3D latent from sparse SDF using the trained SparseSDFVQVAE.

The representation sent to the LLM is the **codebook vector combination** (VQ-VAE output):
we first Encode to discrete indices, then feats = codebook(indices). This is not the raw
encoder output. Mesh reconstruction uses the same encoding_indices: call
vae.Decode(encoding_indices) (and optionally decode_mesh / sparse2mesh).

VAE is used in eval mode and not trained.
"""

from typing import Union, Dict, Tuple
import torch


def _to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch.to(device)


def extract_3d_latent(
    batch: Union[Dict[str, torch.Tensor], "SparseTensor"],
    vae_model: torch.nn.Module,
    device: Union[str, torch.device] = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode 3D sparse SDF with VQ-VAE and return codebook latent (feats, coords).

    Uses Encode() to get discrete indices, then feats = codebook(indices), so the
    returned feats are always codebook vectors, not raw encoder output. For mesh
    reconstruction use extract_3d_latent_and_indices and then vae.Decode(encoding_indices).

    Args:
        batch: Either a dict with keys 'sparse_sdf', 'sparse_index', 'batch_idx'
               (and optional 'factor'), or a SparseTensor from the dataset.
        vae_model: Pre-loaded SparseSDFVQVAE instance (e.g. from trellis.models).
        device: Device to run VAE on.

    Returns:
        feats: [total_N, embed_dim] float, codebook vectors (same as vae.vq.embedding_dim).
        coords: [total_N, 4] long (batch_idx, x, y, z) in latent grid.
    """
    feats, coords, _ = extract_3d_latent_and_indices(batch, vae_model, device)
    return feats, coords


def extract_3d_latent_and_indices(
    batch: Union[Dict[str, torch.Tensor], "SparseTensor"],
    vae_model: torch.nn.Module,
    device: Union[str, torch.device] = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, "SparseTensor"]:
    """
    Encode 3D sparse SDF to indices, then return codebook feats and encoding_indices.

    Single source of truth: Encode(batch) -> encoding_indices; feats = codebook(indices).
    Use encoding_indices with vae.Decode(encoding_indices) for mesh reconstruction.

    Args:
        batch: Either a dict with keys 'sparse_sdf', 'sparse_index', 'batch_idx'
               (and optional 'factor'), or a SparseTensor from the dataset.
        vae_model: Pre-loaded SparseSDFVQVAE instance (e.g. from trellis.models).
        device: Device to run VAE on.

    Returns:
        feats: [total_N, embed_dim] float, codebook vectors.
        coords: [total_N, 4] long (batch_idx, x, y, z) in latent grid.
        encoding_indices: SparseTensor with .feats = indices, .coords = coords;
                          pass to vae.Decode(encoding_indices) to reconstruct.
    """
    vae = vae_model.to(device)
    vae.eval()
    batch = _to_device(batch, device)

    with torch.no_grad():
        encoding_indices = vae.Encode(batch)
        # encoding_indices.feats: [N, 1] float (indices as float from replace())
        indices = encoding_indices.feats.squeeze(-1).long()
        feats = vae.vq.embeddings(indices)
        coords = encoding_indices.coords

    return feats, coords, encoding_indices
