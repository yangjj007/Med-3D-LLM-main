"""
Add mesh tokens to tokenizer and resize + initialize new embeddings in the VL model.
Used for discrete 3D token alignment (no Projector).
"""

from typing import Any, List

import torch
import torch.nn as nn


# Token strings: must match spatial_pool_3d.pooled_sequence_to_mesh_token_string
MESH_START = "<mesh_start>"
MESH_END = "<mesh_end>"
MESH_EMPTY = "<mesh_empty>"
NUM_CODEBOOK = 8192


def get_mesh_token_list() -> List[str]:
    """Return list of all mesh tokens to add: start, end, empty, then <mesh_0> .. <mesh_8191>."""
    tokens = [MESH_START, MESH_END, MESH_EMPTY]
    tokens.extend([f"<mesh_{i}>" for i in range(NUM_CODEBOOK)])
    return tokens


def add_mesh_tokens_to_tokenizer(tokenizer: Any) -> Any:
    """
    Add mesh special tokens to the tokenizer. Call before loading the model
    so that model can be resized to len(tokenizer).
    """
    new_tokens = get_mesh_token_list()
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return tokenizer


def resize_token_embeddings_and_init_mesh(
    model: nn.Module,
    tokenizer: Any,
    num_original_tokens_for_stats: int = 10000,
) -> None:
    """
    Resize the VL model's embedding and lm_head to the new vocab size, then
    initialize the new token embeddings using a mean + small-covariance-noise
    distribution estimated from the original token embeddings.

    Notes:
    - The "full covariance" estimator on the whole vocab is too expensive
      (O(vocab_size * hidden_size^2)). We therefore estimate statistics using
      the first `num_original_tokens_for_stats` rows (configurable) which is
      consistent with the previous implementation's cost profile.
    - For numerical stability, covariance is scaled and jittered.
    """
    if not hasattr(model, "resize_token_embeddings"):
        # Maybe it's the wrapper; try vl_model
        if hasattr(model, "vl_model"):
            model = model.vl_model
        else:
            raise AttributeError("model has no resize_token_embeddings or vl_model")
    old_size = model.get_input_embeddings().weight.shape[0]
    new_size = len(tokenizer)
    if new_size <= old_size:
        return
    model.resize_token_embeddings(new_size)

    emb = model.get_input_embeddings().weight
    device = emb.device
    dtype = emb.dtype
    stats_n = min(int(num_original_tokens_for_stats), int(old_size))
    if stats_n <= 1:
        # Fallback: keep HF default init if we can't estimate stats.
        return

    # Estimate mean/cov on a subset to control cost.
    ref = emb[:stats_n].detach().to(torch.float32)  # stable stats in fp32
    mu = ref.mean(dim=0)  # [d]
    xc = ref - mu
    # Sample covariance (biased, matching user's n divisor); scale down to create "small noise".
    sigma = (xc.T @ xc) / float(stats_n)  # [d,d]
    # Stabilize covariance: PSD jitter and strong scaling
    eps = 1e-6
    cov = (1e-5 * sigma) + (eps * torch.eye(sigma.shape[0], device=sigma.device, dtype=sigma.dtype))

    num_new = int(new_size - old_size)
    if num_new <= 0:
        return

    try:
        from torch.distributions import MultivariateNormal

        dist = MultivariateNormal(mu, covariance_matrix=cov)
        with torch.no_grad():
            samples = dist.sample((num_new,))  # [num_new, d] fp32
            emb[old_size:].copy_(samples.to(device=device, dtype=dtype))
    except Exception:
        # Fallback to diagonal std noise if MVN is unavailable / fails PSD checks.
        var = torch.diag(cov).clamp(min=1e-12)
        std = torch.sqrt(var)
        with torch.no_grad():
            samples = torch.randn((num_new, mu.numel()), device=mu.device, dtype=mu.dtype)
            samples = samples * std + mu
            emb[old_size:].copy_(samples.to(device=device, dtype=dtype))

    # lm_head:
    # - In Qwen3(-VL) HF implementations lm_head is typically tied to embed_tokens,
    #   so initializing embeddings is sufficient.
    # - If not tied (or tying is disabled), align lm_head new rows to embeddings.
    if hasattr(model, "lm_head") and model.lm_head is not None:
        lm = model.lm_head
        if isinstance(lm, nn.Linear) and lm.weight.shape[0] == new_size:
            with torch.no_grad():
                lm.weight[old_size:].copy_(emb[old_size:].detach())
