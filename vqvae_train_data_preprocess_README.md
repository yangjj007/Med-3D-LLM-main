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
python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/ObjaverseXL/raw/hf-objaverse-v1 \
    --output_dir ./train_sdf_dataset \
    --resolutions 512 \
    --filter_aesthetic_score 6.0 \
    --max_workers 1

python dataset_toolkits/sdf_voxelize.py \
    --format trellis500k \
    --input_dir ./TRELLIS-500K/HSSD/raw/objects \
    --output_dir ./train_sdf_dataset \
    --resolutions 512 \
    --filter_aesthetic_score 6.0 \
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


5、todo：对应修改阶段一和二的训练数据读取框架，支持新的训练数据格式