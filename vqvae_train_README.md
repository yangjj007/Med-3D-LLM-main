1、先下载TRELLIS-500Ks：git clone https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K

2、运行build_metadata下载ObjaverseXL的metadata：
python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab   --output_dir TRELLIS-500K/ObjaverseXL

python dataset_toolkits/build_metadata.py HSSD --output_dir TRELLIS-500K/HSSD

3、开始下载
python dataset_toolkits/download.py ObjaverseXL   --output_dir TRELLIS-500K/ObjaverseXL  --rank 0 --world_size 1

python dataset_toolkits/download.py HSSD --output_dir TRELLIS-500K/HSSD --rank 0 --world_size 1

4、对mesh文件进行SDF预处理（✅已完成）

使用新的sdf_voxelize.py脚本进行SDF预处理，支持可配置分辨率，保存为npz格式。

脚本支持2种格式：

1.TRELLIS-500K：
root@9qq3g7rkjocll-0:/yangjunjie/Med-3D-LLM-main/TRELLIS-500K/raw/hf-objaverse-v1# ls
glbs  object-paths.json


其中部分的object-paths.json内容如下，储存了glb文件路径：
{"8476c4170df24cf5bbe6967222d1a42d": "glbs/000-023/8476c4170df24cf5bbe6967222d1a42d.glb", "8ff7f1f2465347cd8b80c9b206c2781e": "glbs/000-
023/8ff7f1f2465347cd8b80c9b206c2781e.glb", "c786b97d08b94d02a1fa3b87d2e86cf1": "glbs/000-023/c786b97d08b94d02a1fa3b87d2e86cf1.glb", ... }

输入路径为：./TRELLIS-500K/raw/hf-objaverse-v1
输出路径为：./train_sdf_dataset  (所有sdf处理后的npz文件扁平储存到输出路径即可，加上个metadata.csv储存caption信息)

root@9qq3g7rkjocll-0:/yangjunjie/Med-3D-LLM-main/TRELLIS-500K# ls
3D-FUTURE.csv  ABO.csv  HSSD.csv  ObjaverseXL_github.csv  ObjaverseXL_sketchfab.csv  README.md  Toys4k.csv  merged_records  metadata.csv  raw  statistics.txt

这是部分的metadata.csv内容（不是每一条文件我都下载了的，不存在的就跳过，也不需要储存metadata）：

file_identifier   sha256   aesthetic_score   captions
c1995854d46ecaa525e10  849afb22b3760d3c73a31  6.554802894592285
333b72b6f481771cca4277 c48b8a3bfae6d043e331b  5.946990013122559  ns.", "Armored warrior."]
6e7a522bc6b227a1781a3  9470b9b4f5bd1e202e598  6.2495198249816895   booth.", "Donut booth."]
e495db051d5b47d001a32  724505899e52ed86dff21  6.868131637573242   je futuristic motorcycle."]


2.我自己创建的带部件级label的数据：


## 目录结构概览

```text
dataset_root/
├── metadata.csv          # 核心索引文件，包含所有模型的元数据和标注
├── images/               # 存放所有截图的文件夹
│   ├── <sha256_id>/      # 每个模型单独一个子文件夹，避免文件名冲突
│   │   ├── overall/      # 整体视图 (main.png, top.png 等)
│   │   └── materials/    # 材质细节截图 (保留原始特殊字符文件名)
│   └── ...
├── <sha256_id>.glb       # 3D 模型文件 (扁平放置)
├── <sha256_id>.glb
└── ...

CSV 文件包含以下列：

字段名,说明
sha256,模型的唯一标识符 (ID)
glb_file,GLB 模型文件的相对文件名
overall_label,模型的整体描述/标签 (来自 info.json)
materials_captions,材质部分的详细描述 (JSON 格式字符串，包含部位名称和对应文本)
image_dir_path,该模型图片所在的相对路径 (例如 images/<id>/)
view_keys,"该模型包含的视图列表 (如 axial, main, top, side)"

## 使用sdf_voxelize.py进行SDF预处理

### 处理TRELLIS-500K数据：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/ObjaverseXL/raw/hf-objaverse-v1 \
    --output_dir ./train_sdf_dataset/res512_thre0.5 \
    --filter_aesthetic_score 6.5 \
    --resolutions 512 \
    --threshold_factor 0.5 \
    --max_workers 1


CUDA_VISIBLE_DEVICES=1 python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/HSSD/raw/objects \
    --output_dir ./train_sdf_dataset/res512_thre0.1 \
    --filter_aesthetic_score 4.0 \
    --resolutions 512 \
    --threshold_factor 0.1 \
    --max_workers 1
```

### 处理自定义标注数据：
```bash
python dataset_toolkits/sdf_voxelize.py \
    --format custom \
    --input_dir ./my_dataset \
    --output_dir ./train_sdf_dataset \
    --resolutions 512 \
    --max_workers 1
```


5、使用sdf_voxelize输出的数据进行阶段一和阶段二训练（✅已支持）

SparseSDF数据集类已适配sdf_voxelize.py的输出格式：
- 支持扁平目录布局：`{output_dir}/{sha256}_r{resolution}.npz`
- 支持metadata.csv中的`sdf_computed`列名
- 兼容旧版compute_sparse_sdf.py的子目录布局

### 阶段1训练 - 冻结VAE，训练码本
```bash
export ATTN_BACKEND=xformers
python train.py \
    --config configs/vae/sdf_vqvae_stage1.json \
    --output_dir outputs_pad_sdf/sdf_vqvae_stage1_512_0.5 \
    --data_dir ./train_sdf_dataset/res512_thre0.5/ \
    --num_gpus 4 2>&1 | tee train1.log
```

### 阶段2训练 - 联合微调
与阶段3一致：训练前向使用与推理 `Decode()` 相同的满块展开（`train_full_block_decode: true`），并对满块展开产生、但不在输入 sparse SDF 中的输出坐标施加与 meshing dense fill 一致的 `1.0` 约束（`lambda_unmatched_output`）。

**Checkpoint 数量（磁盘）**：在 `trainer.args` 中可设 `save_total_limit`（默认 `3`）。每次按 `i_save` 保存后，会在 `output_dir/ckpts/` 下只保留**步数最新**的 N 套完整权重（含 `misc_step*.pt`、各模型 `*_step*.pt`、EMA `*_ema*_step*.pt`），更早的 step 整批删除。设为 `0` 表示不限制、不删旧文件。

**空间下采样倍数与通道渐进（默认 `extra_down_up_levels: 0`）**：`SparseSDFVQVAE` 的 encoder 在输入体素网格上依次做 **4 次** `×2` 空间下采样（共 **×16**），通道序列为 **C/16 → C/8 → C/4 → C/2 → C**（例如 C=512 时即 **32→64→128→256→512**）；decoder 对称地为 **C → C/2 → C/4 → C/8 → C/16** 四级 subdivide 再上采样回原分辨率。因此当 `models.vqvae.args.resolution` 为瓶颈网格边长（如 **32**）且 SDF 数据为 **512³** 时，整体为 **512→32** 的 latent 瓶颈。从旧版仅 **3 级（×8）** 的 checkpoint 继续训练时，`load_pretrained_vae` 会按**张量形状**加载兼容子层，并对 decoder 做 **3→4 块** 的键重映射（旧 `upsample.1/2` → 新 `upsample.2/3`）；**不兼容的块保持随机初始化**，需充分微调。

**扩展上下采样（`models.vqvae.args.extra_down_up_levels`）**：当该值 **L>0** 时，在以上 **4 级渐进** 之外，再在 encoder **末尾**追加 **L** 个 **512→512** 下采样块、在 decoder **开头**追加 **L** 个 **512→512** subdivide 块（再进入上述四级），空间倍数再乘 **2^L**。此时与仅 4 级渐进或旧结构的 checkpoint 均可能不完全对齐；首次用旧权重初始化时，请使用**新的 `output_dir`**，并确保 `load_dir` 下**没有**可恢复的 `ckpts/misc_*.pt`（否则会走完整 resume，`optimizer` 与旧结构不兼容）。新结构产生自己的 `ckpts` 之后，再在同一目录用 `--ckpt latest` 恢复即可。

```bash
export ATTN_BACKEND=xformers
python train.py \
    --config configs/vae/sdf_vqvae_stage2.json \
    --output_dir outputs_pad_sdf/sdf_vqvae_stage2_512_0.5 \
    --data_dir ./train_sdf_dataset \
    --load_dir outputs_pad_sdf/sdf_vqvae_stage2_512_0.5 \
    --ckpt latest \
    --num_gpus 4 2>&1 | tee train2.log
```

export ATTN_BACKEND=xformers
python train.py \
    --config configs/vae/sdf_vqvae_stage2.json \
    --output_dir outputs_pad_sdf/sdf_vqvae_no_cat_16sample_512_0.5 \
    --data_dir ./train_sdf_dataset \
    --load_dir outputs_pad_sdf/sdf_vqvae_no_cat_16sample_512_0.5 \
    --ckpt latest \
    --num_gpus 4 2>&1 | tee train2.log

export ATTN_BACKEND=xformers
python train.py \
    --config configs/vae/sdf_vqvae_stage2.json \
    --output_dir outputs_pad_sdf/sdf_vqvae_256_0.5 \
    --data_dir ./train_sdf_dataset_256 \
    --load_dir outputs_pad_sdf/sdf_vqvae_256_0.5 \
    --ckpt latest \
    --num_gpus 4 2>&1 | tee train2.log

### 阶段3训练 - 冻结Encoder和码本，只训练Decoder
阶段3会加载阶段2权重，并在训练时使用与推理 `Decode()` 一致的满块展开（`train_full_block_decode: true`）。额外由满块展开产生、但不在输入 sparse SDF 中的输出坐标，会按 meshing 的 dense fill 约定约束到 `1.0`。

```bash
export ATTN_BACKEND=xformers
python train.py \
    --config configs/vae/sdf_vqvae_stage3.json \
    --output_dir outputs_pad_sdf/sdf_vqvae_stage3_512_0.5 \
    --data_dir ./train_sdf_dataset/res512_thre0.5/ \
    --num_gpus 3 2>&1 | tee train3.log
```

```bash
    --load_dir outputs_pad_sdf/sdf_vqvae_stage2_512_0.5 \
    --ckpt latest \
```

注意：`--data_dir` 应指向 sdf_voxelize.py 的 `--output_dir`（即 `./train_sdf_dataset`）