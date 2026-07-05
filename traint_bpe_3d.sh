#!/usr/bin/env bash
python dataset_toolkits/train_bpe_3d.py \
--sdf_dir ./train_sdf_dataset/res512_thre0.5 \
--vae_config ./configs/vae/sdf_vqvae_stage2.json \
--vae_ckpt ./outputs/sdf_vqvae_stage2_512_0.5-amp/ckpts/vqvae_step0000100.pt \
--corpus_cache ./bpe_corpus.npz \
--out_merge_table ./merge_table.json
