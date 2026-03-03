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
    num_original_tokens_for_stats: int = 1000,
) -> None:
    """
    Resize the VL model's embedding and lm_head to the new vocab size, then
    initialize the new mesh token embeddings using mean and std of the first
    num_original_tokens_for_stats tokens from the original vocab.
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
    # Initialize new token rows with mean/std of original embeddings
    emb = model.get_input_embeddings().weight
    ref = emb[:num_original_tokens_for_stats].detach()
    mean = ref.mean(dim=0)
    std = ref.std(dim=0).clamp(min=1e-6)
    with torch.no_grad():
        emb[old_size:].normal_(mean=0, std=1)
        emb[old_size:].mul_(std).add_(mean)
    # lm_head if exists (tied or not)
    if hasattr(model, "lm_head") and model.lm_head is not None:
        lm = model.lm_head
        if isinstance(lm, nn.Linear) and lm.weight.shape[0] == new_size:
            ref_lm = lm.weight[:num_original_tokens_for_stats].detach()
            mean_lm = ref_lm.mean(dim=0)
            std_lm = ref_lm.std(dim=0).clamp(min=1e-6)
            with torch.no_grad():
                lm.weight[old_size:].normal_(mean=0, std=1)
                lm.weight[old_size:].mul_(std_lm).add_(mean_lm)
