# VAE + Qwen3-VL integration for 3D understanding and mesh reconstruction.
# Uses trained SparseSDFVQVAE: Encode -> indices -> codebook(indices) = feats [N, embed_dim];
# feats are projected to LLM hidden size and prepended to text. Same encoding_indices
# can be passed to vae.Decode(encoding_indices) for mesh reconstruction.

from .vae_latent_extractor import extract_3d_latent, extract_3d_latent_and_indices
from .sequence_3d import prepare_3d_sequence
from .projector import Projector3D, PositionEncoder3D
from .model import Qwen3VLWith3DBranch
from .tokenizer_3d import (
    get_mesh_token_list,
    add_mesh_tokens_to_tokenizer,
    resize_token_embeddings_and_init_mesh,
)
from .spatial_pool_3d import (
    MESH_EMPTY,
    GRID_SIZE,
    POOLED_LEN,
    spatial_pool_8x8x8,
    encoding_indices_to_pooled_sequence,
    pooled_sequence_to_mesh_token_string,
    batch_encoding_indices_to_pooled_sequences,
)
from .variable_length_3d import (
    DEFAULT_MAX_SAFE_LENGTH,
    encoding_indices_to_variable_length_sequence,
    variable_length_sequence_to_mesh_token_string,
    batch_encoding_indices_to_variable_length_sequences,
)

__all__ = [
    "extract_3d_latent",
    "extract_3d_latent_and_indices",
    "prepare_3d_sequence",
    "Projector3D",
    "PositionEncoder3D",
    "Qwen3VLWith3DBranch",
    "MESH_EMPTY",
    "GRID_SIZE",
    "POOLED_LEN",
    "spatial_pool_8x8x8",
    "encoding_indices_to_pooled_sequence",
    "pooled_sequence_to_mesh_token_string",
    "batch_encoding_indices_to_pooled_sequences",
    "get_mesh_token_list",
    "add_mesh_tokens_to_tokenizer",
    "resize_token_embeddings_and_init_mesh",
    "DEFAULT_MAX_SAFE_LENGTH",
    "encoding_indices_to_variable_length_sequence",
    "variable_length_sequence_to_mesh_token_string",
    "batch_encoding_indices_to_variable_length_sequences",
]
