"""
端到端烟雾测试：验证 3D-VL 整条流程可跑通（VAE Encode -> 码本 -> projector -> LLM forward / generate，以及 Decode 重建）。

用法（项目根目录）：
  # 快速模式：不下载 Qwen2-VL，用轻量 mock 测 3D 分支 + VAE Encode/Decode
  python vae_qwen3vl/run_smoke_test.py --vae_config configs/vae/sdf_vqvae_stage1.json --quick

  # 完整模式：加载真实 Qwen2-VL，跑 forward + Decode（需网络与显存）
  python vae_qwen3vl/run_smoke_test.py --vae_config configs/vae/sdf_vqvae_stage1.json

  # 若有已训练 VAE 权重，可指定以验证与 checkpoint 兼容
  python vae_qwen3vl/run_smoke_test.py --vae_config configs/vae/sdf_vqvae_stage1.json --vae_ckpt path/to/vae.pt --quick
"""

import argparse
import os
import sys
import json

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _make_dummy_inputs_3d(device, n_points=200, resolution=64):
    """构造满足 SparseSDFVQVAE 输入的 dummy batch（dict）。"""
    return {
        "sparse_sdf": torch.randn(n_points, 1, device=device, dtype=torch.float32),
        "sparse_index": torch.randint(0, resolution, (n_points, 3), device=device),
        "batch_idx": torch.zeros(n_points, dtype=torch.long, device=device),
    }


def _build_fake_vae(device: str, n_embed=8192, embed_dim=16):
    """最小假 VAE：仅用于在不依赖 trellis 时跑通 3D 分支与 projector。"""
    class _FakeSparseTensor:
        def __init__(self, feats, coords):
            self.feats = feats
            self.coords = coords

    class _FakeVQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(n_embed, embed_dim)
            self.num_embeddings = n_embed

    class _FakeVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.vq = _FakeVQ()

        def Encode(self, batch):
            n = batch["sparse_sdf"].shape[0]
            device = batch["sparse_sdf"].device
            indices = torch.randint(0, n_embed, (n,), device=device)
            coords = torch.cat([
                batch["batch_idx"].unsqueeze(1),
                batch["sparse_index"],
            ], dim=1)
            return _FakeSparseTensor(indices.unsqueeze(1).float(), coords)

        def Decode(self, encoding_indices):
            idx = encoding_indices.feats.squeeze(-1).long()
            feats = self.vq.embeddings(idx)
            return _FakeSparseTensor(feats, encoding_indices.coords)

    return _FakeVAE().to(device).eval()


def _build_vae(vae_config_path: str, vae_ckpt_path, device: str, use_fake_if_fail: bool = True):
    """从 config 实例化 VAE；若提供 ckpt 则加载，否则随机初始化。失败时若 use_fake_if_fail 则退回假 VAE。"""
    try:
        from trellis.models import SparseSDFVQVAE
    except Exception as e:
        if use_fake_if_fail:
            print(f"[run_smoke_test] trellis 未安装或依赖缺失 ({e})，使用假 VAE 仅验证流程与形状。")
            with open(vae_config_path, "r") as f:
                cfg = json.load(f)
            args = cfg["models"]["vqvae"]["args"]
            embed_dim = args.get("latent_channels", 16)
            n_embed = args.get("num_embeddings", 8192)
            return _build_fake_vae(device, n_embed=n_embed, embed_dim=embed_dim)
        raise
    with open(vae_config_path, "r") as f:
        config = json.load(f)
    vae_args = config["models"]["vqvae"]["args"]
    model = SparseSDFVQVAE(**vae_args)
    if vae_ckpt_path and os.path.isfile(vae_ckpt_path):
        ckpt = torch.load(vae_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    return model


def _build_mock_vl(hidden_size: int = 64, vocab_size: int = 152064):
    """构造与 Qwen2-VL 接口兼容的轻量 mock（forward 返回 loss，generate 返回 token ids）。"""
    _hs = hidden_size
    class _Config:
        hidden_size = _hs
        text_config = type("_T", (), {"hidden_size": _hs})()

    class _MockVL(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Config()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
            loss = inputs_embeds.float().mean()
            if labels is not None:
                loss = loss + 0.0 * labels.float().mean()
            return {"loss": loss}


        def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=2, do_sample=False, pad_token_id=0, **kwargs):
            b, l, _ = inputs_embeds.shape
            return torch.zeros(b, l + max_new_tokens, dtype=torch.long, device=inputs_embeds.device)

    return _MockVL()


def run_quick(device: str, vae_config: str, vae_ckpt, max_3d_tokens: int):
    """快速模式：mock VL + 真实或假 VAE，测 3D 分支与 Decode。"""
    from vae_qwen3vl import Qwen3VLWith3DBranch, extract_3d_latent_and_indices

    vae = _build_vae(vae_config, vae_ckpt, device, use_fake_if_fail=True)
    mock_vl = _build_mock_vl(hidden_size=64).to(device)
    model = Qwen3VLWith3DBranch(
        vae_model=vae,
        max_3d_tokens=max_3d_tokens,
        use_3d_pos=False,
        use_vl_model=mock_vl,
    )
    model = model.to(device)
    model.eval()

    inputs_3d = _make_dummy_inputs_3d(device, n_points=200)
    batch_size = 1
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = input_ids.clone()
    labels[:, :-1] = -100

    # 1) forward_with_3d(inputs_3d=...)
    with torch.no_grad():
        out = model.forward_with_3d(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_3d=inputs_3d,
        )
    assert hasattr(out, "loss") or "loss" in out, "forward_with_3d should return loss"
    loss = getattr(out, "loss", None) or out.get("loss")
    assert loss.dim() == 0, "loss should be scalar"
    assert "encoding_indices_3d" in out, "outputs should contain encoding_indices_3d"
    encoding_indices = out["encoding_indices_3d"]

    # 2) Decode(encoding_indices)
    with torch.no_grad():
        recon = vae.Decode(encoding_indices)
    assert hasattr(recon, "feats") and hasattr(recon, "coords"), "Decode should return SparseTensor-like"
    assert recon.feats.shape[0] == encoding_indices.feats.shape[0], "recon length should match indices"

    # 3) get_3d_embeds_and_encoding_indices
    embeds_3d, mask_3d, enc_idx = model.get_3d_embeds_and_encoding_indices(inputs_3d, device=device)
    assert embeds_3d.dim() == 3 and embeds_3d.shape[0] == 1, "embeds_3d should be [1, seq_3d, H]"
    assert mask_3d.shape[0] == 1 and mask_3d.shape[1] == embeds_3d.shape[1], "mask_3d length should match"
    assert enc_idx.feats.shape[0] == encoding_indices.feats.shape[0], "indices from get_3d_embeds should match"

    # 4) extract_3d_latent / extract_3d_latent_and_indices
    feats, coords, enc_indices2 = extract_3d_latent_and_indices(inputs_3d, vae, device=device)
    assert feats.shape[1] == vae.vq.embeddings.weight.shape[1], "feats dim should match codebook"
    assert coords.shape[1] == 4, "coords should be [N, 4]"

    print("[Quick] All checks passed: forward_with_3d, encoding_indices_3d, Decode, get_3d_embeds_and_encoding_indices, extract_3d_latent_and_indices.")
    return True


def run_full(device: str, vae_config: str, vae_ckpt, vl_model: str, max_3d_tokens: int):
    """完整模式：真实 VAE + 真实 Qwen2-VL，跑 forward + Decode。"""
    from vae_qwen3vl import Qwen3VLWith3DBranch

    vae = _build_vae(vae_config, vae_ckpt, device, use_fake_if_fail=False)
    model = Qwen3VLWith3DBranch(
        model_name_or_path=vl_model,
        vae_model=vae,
        max_3d_tokens=max_3d_tokens,
        use_3d_pos=False,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()

    inputs_3d = _make_dummy_inputs_3d(device, n_points=300)
    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = input_ids.clone()
    labels[:, :-1] = -100

    with torch.no_grad():
        out = model.forward_with_3d(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_3d=inputs_3d,
        )
    loss = getattr(out, "loss", None) or out.get("loss")
    assert loss is not None, "full forward should return loss"
    assert "encoding_indices_3d" in out, "outputs should contain encoding_indices_3d"

    encoding_indices = out["encoding_indices_3d"]
    with torch.no_grad():
        recon = vae.Decode(encoding_indices)
    assert recon.feats.shape[0] == encoding_indices.feats.shape[0]

    print("[Full] All checks passed: forward_with_3d with real VL, encoding_indices_3d, Decode.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test 3D-VL pipeline")
    parser.add_argument("--vae_config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "vae", "sdf_vqvae_stage1.json"))
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--vl_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--max_3d_tokens", type=int, default=512)
    parser.add_argument("--quick", action="store_true", help="Use mock VL and small data; no download.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isfile(args.vae_config):
        print(f"VAE config not found: {args.vae_config}")
        sys.exit(1)

    try:
        if args.quick:
            run_quick(device, args.vae_config, args.vae_ckpt, args.max_3d_tokens)
        else:
            run_full(device, args.vae_config, args.vae_ckpt, args.vl_model, args.max_3d_tokens)
        print("Smoke test OK.")
    except Exception as e:
        print(f"Smoke test failed: {e}")
        raise


if __name__ == "__main__":
    main()
