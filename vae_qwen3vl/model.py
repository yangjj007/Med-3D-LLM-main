"""
Wrapper that adds a 3D branch to Qwen2-VL / Qwen3-VL: VAE (frozen) + projector
produce 3D token embeddings that are prepended to text and fed to the LLM.
"""

from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn

from .projector import Projector3D
from .sequence_3d import prepare_3d_sequence, prepare_3d_sequence_batched
from .vae_latent_extractor import extract_3d_latent, extract_3d_latent_and_indices


def _get_vl_model_and_config(model_name_or_path: str, **kwargs):
    try:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
        model_cls = Qwen3VLForConditionalGeneration
        config_cls = Qwen3VLConfig
    except Exception:
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig
        model_cls = Qwen2VLForConditionalGeneration
        config_cls = Qwen2VLConfig
    model = model_cls.from_pretrained(model_name_or_path, **kwargs)
    return model, model.config


class Qwen3VLWith3DBranch(nn.Module):
    """
    Qwen2-VL or Qwen3-VL with an added 3D branch. For 3D input: VAE encode ->
    prepare_3d_sequence -> projector -> 3D token embeddings, then prepended to
    text embeddings and fed to the LLM.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        vae_model: Optional[nn.Module] = None,
        latent_dim: int = 16,
        max_3d_tokens: int = 2048,
        use_3d_pos: bool = False,
        projector_num_layers: int = 1,
        torch_dtype: Optional[torch.dtype] = None,
        use_vl_model: Optional[nn.Module] = None,
        **vl_kwargs,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.max_3d_tokens = max_3d_tokens
        if torch_dtype is None:
            torch_dtype = torch.float32
        if use_vl_model is not None:
            self.vl_model = use_vl_model
            cfg = getattr(use_vl_model, "config", None)
            if cfg is None:
                cfg = type("_C", (), {"hidden_size": 1024, "text_config": type("_T", (), {"hidden_size": 1024})()})()
            self.vl_config = cfg
        else:
            self.vl_model, self.vl_config = _get_vl_model_and_config(
                model_name_or_path, torch_dtype=torch_dtype, **vl_kwargs
            )
        hidden_size = getattr(
            self.vl_config, "hidden_size", None
        ) or getattr(
            getattr(self.vl_config, "text_config", None), "hidden_size", 1024
        )
        if vae_model is not None and hasattr(vae_model, "vq") and hasattr(vae_model.vq, "embeddings"):
            latent_dim = vae_model.vq.embeddings.weight.shape[1]
        self.projector = Projector3D(
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            num_layers=projector_num_layers,
            use_3d_pos=use_3d_pos,
            pos_mode="sinusoidal",
            max_coord=64,
        )
        if vae_model is not None:
            for p in vae_model.parameters():
                p.requires_grad = False

    def get_3d_embeds(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        attention_mask_3d: Optional[torch.Tensor] = None,
        coords_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D latent to LLM hidden size.
        feats: [N, 16] or [B, N, 16]; coords: [N, 4] or [B, N, 4].
        Returns: embeddings [seq_len, hidden_size] or [B, seq_len, hidden_size],
                 attention_mask [seq_len] or [B, seq_len].
        """
        if feats.dim() == 2:
            feats, attention_mask_3d, coords_out = prepare_3d_sequence(
                feats, coords, max_3d_tokens=self.max_3d_tokens
            )
            feats = feats.unsqueeze(0)
            attention_mask_3d = attention_mask_3d.unsqueeze(0)
            coords_out = coords_out.unsqueeze(0)
        embeds = self.projector(feats, coords_out)
        return embeds, attention_mask_3d

    def get_3d_embeds_and_encoding_indices(
        self,
        inputs_3d: Dict[str, torch.Tensor],
        device: Union[str, torch.device] = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Get 3D token embeddings and encoding_indices for mesh reconstruction.

        Use encoding_indices with self.vae_model.Decode(encoding_indices) to reconstruct
        sparse SDF, or decode_mesh / sparse2mesh for mesh output.

        Args:
            inputs_3d: Batch dict with 'sparse_sdf', 'sparse_index', 'batch_idx'.
            device: Device for VAE and tensors.

        Returns:
            embeds_3d: [1, seq_len, hidden_size] 3D token embeddings.
            attention_mask_3d: [1, seq_len] mask for 3D tokens.
            encoding_indices: SparseTensor to pass to vae_model.Decode(...).
        """
        if self.vae_model is None:
            raise ValueError("vae_model is required for get_3d_embeds_and_encoding_indices.")
        feats_3d, coords_3d, encoding_indices = extract_3d_latent_and_indices(
            inputs_3d, self.vae_model, device=device
        )
        feats_seq, mask_3d, coords_seq = prepare_3d_sequence(
            feats_3d, coords_3d, max_3d_tokens=self.max_3d_tokens
        )
        feats_seq = feats_seq.unsqueeze(0).to(device)
        mask_3d = mask_3d.unsqueeze(0).to(device)
        coords_seq = coords_seq.unsqueeze(0).to(device)
        embeds_3d = self.projector(feats_seq, coords_seq)
        return embeds_3d, mask_3d, encoding_indices

    def forward_with_3d(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_3d: Optional[Dict[str, torch.Tensor]] = None,
        feats_3d: Optional[torch.Tensor] = None,
        coords_3d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward when input includes 3D. Either pass inputs_3d (batch dict for VAE)
        or precomputed feats_3d [B, N, latent_dim] and coords_3d [B, N, 4].
        When inputs_3d + vae_model are used, outputs include encoding_indices_3d
        for mesh reconstruction via vae_model.Decode(encoding_indices_3d).
        """
        encoding_indices = None
        if inputs_3d is not None and self.vae_model is not None:
            feats_3d, coords_3d, encoding_indices = extract_3d_latent_and_indices(
                inputs_3d, self.vae_model, device=input_ids.device
            )
        elif inputs_3d is None and (feats_3d is None or coords_3d is None):
            raise ValueError("Provide either inputs_3d (with vae_model) or feats_3d and coords_3d.")
        if feats_3d is None or coords_3d is None:
            raise ValueError("Provide either inputs_3d (with vae_model) or feats_3d and coords_3d.")

        batch_size = input_ids.shape[0]
        if feats_3d.dim() == 2:
            feats_seq, mask_3d, coords_seq = prepare_3d_sequence_batched(
                feats_3d, coords_3d, batch_size=batch_size, max_3d_tokens=self.max_3d_tokens
            )
        else:
            if feats_3d.dim() == 3:
                feats_flat = feats_3d.reshape(-1, feats_3d.shape[-1])
                b_idx = torch.arange(batch_size, device=feats_3d.device).repeat_interleave(feats_3d.shape[1])
                coords_flat = coords_3d.reshape(-1, 4)
                coords_flat = torch.cat([b_idx.unsqueeze(1).to(coords_flat.dtype), coords_flat[:, 1:4]], dim=1)
                feats_3d, coords_3d = feats_flat, coords_flat
            feats_seq, mask_3d, coords_seq = prepare_3d_sequence_batched(
                feats_3d, coords_3d, batch_size=batch_size, max_3d_tokens=self.max_3d_tokens
            )
        if feats_seq.device != input_ids.device:
            feats_seq = feats_seq.to(input_ids.device)
            mask_3d = mask_3d.to(input_ids.device)
            coords_seq = coords_seq.to(input_ids.device)
        embeds_3d = self.projector(feats_seq, coords_seq)

        embed_tokens = self.vl_model.get_input_embeddings()
        text_embeds = embed_tokens(input_ids)
        embeds_3d = embeds_3d.to(text_embeds.dtype)
        seq_3d = embeds_3d.shape[1]
        combined_embeds = torch.cat([embeds_3d, text_embeds], dim=1)
        if attention_mask is not None:
            combined_attention_mask = torch.cat(
                [mask_3d.to(attention_mask.dtype), attention_mask], dim=1
            )
        else:
            combined_attention_mask = torch.cat(
                [mask_3d.to(text_embeds.dtype), torch.ones_like(input_ids, dtype=text_embeds.dtype, device=input_ids.device)], dim=1
            )

        if labels is not None:
            pad_labels = torch.full(
                (labels.shape[0], seq_3d),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([pad_labels, labels], dim=1)
        else:
            combined_labels = None
        outputs = self.vl_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
        )
        if encoding_indices is not None:
            outputs["encoding_indices_3d"] = encoding_indices
        return outputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_3d: Optional[Dict[str, torch.Tensor]] = None,
        feats_3d: Optional[torch.Tensor] = None,
        coords_3d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Any]:
        if inputs_3d is not None or feats_3d is not None:
            return self.forward_with_3d(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_3d=inputs_3d,
                feats_3d=feats_3d,
                coords_3d=coords_3d,
            )
        return self.vl_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs,
        )

    def generate(self, **kwargs):
        return self.vl_model.generate(**kwargs)
