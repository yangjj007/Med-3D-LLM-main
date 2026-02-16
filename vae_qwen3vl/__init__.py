# VAE + Qwen3-VL integration for 3D understanding and mesh reconstruction.
# Uses trained SparseSDFVQVAE: Encode -> indices -> codebook(indices) = feats [N, embed_dim];
# feats are projected to LLM hidden size and prepended to text. Same encoding_indices
# can be passed to vae.Decode(encoding_indices) for mesh reconstruction.

from .vae_latent_extractor import extract_3d_latent, extract_3d_latent_and_indices
from .sequence_3d import prepare_3d_sequence
from .projector import Projector3D, PositionEncoder3D
from .model import Qwen3VLWith3DBranch

__all__ = [
    "extract_3d_latent",
    "extract_3d_latent_and_indices",
    "prepare_3d_sequence",
    "Projector3D",
    "PositionEncoder3D",
    "Qwen3VLWith3DBranch",
]
