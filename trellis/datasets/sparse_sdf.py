"""
Sparse SDF dataset for training VQVAE models.
Loads pre-computed sparse SDF data from .npz files.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple

from .components import StandardDatasetBase

# One-time warning flag so we don't spam the log when loading legacy npz files
_EDGE_MASK_WARN_ISSUED = False


def _resolve_npz_path(root: str, instance: str, resolution: int) -> Optional[str]:
    """Flat layout first, then sparse_sdf/ (legacy)."""
    flat = os.path.join(root, f"{instance}_r{resolution}.npz")
    if os.path.exists(flat):
        return flat
    sub = os.path.join(root, "sparse_sdf", f"{instance}_r{resolution}.npz")
    if os.path.exists(sub):
        return sub
    return None


def _npz_sparse_index_touches_edge(path: str, resolution: int, edge_margin: int) -> bool:
    """
    True iff any voxel lies in the boundary band [0, m) or [resolution-m, resolution).
    Uses mmap so only the sparse_index array is paged in (not sparse_sdf).
    """
    m = edge_margin
    ub = resolution - edge_margin
    with np.load(path, mmap_mode="r") as z:
        idx = z["sparse_index"]
        if np.any(idx < m) or np.any(idx >= ub):
            return True
    return False


def _edge_margin_scan_one(
    args: Tuple[int, str, str, int, int],
) -> Tuple[int, bool, Optional[str]]:
    """Returns (index, keep_instance, error_msg)."""
    i, root, instance, resolution, edge_margin = args
    path = _resolve_npz_path(root, instance, resolution)
    if path is None:
        return i, False, "npz_missing"
    try:
        bad = _npz_sparse_index_touches_edge(path, resolution, edge_margin)
        return i, not bad, None
    except Exception as e:
        return i, False, str(e)


def _npz_count_encoder_band_voxels(path: str, thresh: float) -> int:
    """
    mmap 读取 sparse_sdf，统计归一化 |SDF| <= thresh 的体素数（与训练 enc_mask 一致）。
    用于剔除「壳层全在紧带之外」的异常 npz（例如量化后 min|sdf|>thresh）。
    """
    with np.load(path, mmap_mode="r") as z:
        s = z["sparse_sdf"]
        if s.ndim == 2 and s.shape[1] >= 1:
            a = np.abs(np.asarray(s[:, 0], dtype=np.float32))
        else:
            a = np.abs(np.asarray(s, dtype=np.float32).reshape(-1))
    return int(np.count_nonzero(a <= float(thresh)))


def _encoder_band_scan_one(
    args: Tuple[int, str, str, int, float, int],
) -> Tuple[int, bool, Optional[str]]:
    """Returns (index, keep_instance, error_msg)."""
    i, root, instance, resolution, thresh, min_vox = args
    path = _resolve_npz_path(root, instance, resolution)
    if path is None:
        return i, False, "npz_missing"
    try:
        c = _npz_count_encoder_band_voxels(path, thresh)
        return i, c >= min_vox, None
    except Exception as e:
        return i, False, str(e)


class SparseSDF(StandardDatasetBase):
    """
    Sparse SDF dataset for VQVAE training.
    
    Each instance is stored as {sha256}_r{resolution}.npz containing:
        - sparse_sdf: SDF values [N, 1]
        - sparse_index: 3D coordinates [N, 3]
        - resolution: Grid resolution
    
    Args:
        roots (str): Comma-separated paths to dataset directories
        resolution (int): Grid resolution to load (default: 64)
        min_points (int): Minimum number of sparse points (default: 10)
        max_points (int): Maximum number of sparse points, will subsample if exceeded (default: None)
        edge_margin (int): Skip samples containing voxels too close to the spatial boundary.
        edge_scan_workers (int, optional): Parallel threads for one-time edge scan at init.
            Default uses min(32, cpu_count). Set to 1 to disable parallelism.
        input_band_factor (float): Must match trainer; used with preprocessing_extra_band_factor
            to set encoder tight-band threshold in normalized SDF units.
        preprocessing_extra_band_factor (float): Must match trainer preprocessing.
        prefilter_encoder_band (bool): If True, one-time mmap scan at init drops npz with
            fewer than ``min_encoder_band_voxels`` points satisfying |sdf| <= threshold.
        min_encoder_band_voxels (int): Minimum count of voxels in encoder input band (default 1).
        encoder_band_scan_workers (int, optional): Threads for encoder-band prefilter; defaults
            to edge_scan_workers logic when None.
        snapshot_mc_threshold (float): Marching Cubes iso-level for training snapshot images
            (default 0.0, same as eval ``mc_threshold`` for zero-crossing of signed SDF).
    """
    
    def __init__(
        self,
        roots: str,
        resolution: int = 64,
        min_points: int = 10,
        max_points: int = None,
        edge_margin: int = 0,
        edge_scan_workers: Optional[int] = None,
        input_band_factor: float = 0.5,
        preprocessing_extra_band_factor: float = 4.0,
        prefilter_encoder_band: bool = True,
        min_encoder_band_voxels: int = 1,
        encoder_band_scan_workers: Optional[int] = None,
        snapshot_mc_threshold: float = 0.0,
    ):
        self.resolution = resolution
        self.min_points = min_points
        self.max_points = max_points
        self.edge_margin = edge_margin
        self.edge_scan_workers = edge_scan_workers
        self.input_band_factor = float(input_band_factor)
        self.preprocessing_extra_band_factor = float(preprocessing_extra_band_factor)
        self.snapshot_mc_threshold = float(snapshot_mc_threshold)
        self.encoder_input_sdf_thresh = (
            self.input_band_factor / self.preprocessing_extra_band_factor
        )
        self.prefilter_encoder_band = bool(prefilter_encoder_band)
        self.min_encoder_band_voxels = int(min_encoder_band_voxels)
        self.encoder_band_scan_workers = encoder_band_scan_workers
        self.value_range = (0, 1)  # For visualization
        
        if self.edge_margin > 0 and 2 * self.edge_margin >= self.resolution:
            raise ValueError(
                f"edge_margin={self.edge_margin} 与 resolution={self.resolution} 不兼容："
                f"需满足 2*edge_margin < resolution，否则不存在完全落在内部的体素网格。"
            )

        super().__init__(roots)

        if self.edge_margin > 0:
            self._prefilter_edge_margin_instances()

        if self.prefilter_encoder_band:
            self._prefilter_encoder_band_instances()
    
    def filter_metadata(self, metadata):
        """Filter metadata to only include instances with sparse SDF computed.
        
        Supports both column naming conventions:
        - 'sdf_computed': from sdf_voxelize.py output
        - 'sparse_sdf_computed': from compute_sparse_sdf.py output (legacy)
        """
        stats = {}
        
        # Check for SDF computed flag (support both naming conventions)
        if 'sdf_computed' in metadata.columns:
            metadata = metadata[metadata['sdf_computed'] == True]
            stats['SDF computed'] = len(metadata)
        elif 'sparse_sdf_computed' in metadata.columns:
            metadata = metadata[metadata['sparse_sdf_computed'] == True]
            stats['Sparse SDF computed'] = len(metadata)
        
        # Check if resolution-specific point count exists
        points_col = f'r{self.resolution}_num_points'
        if points_col in metadata.columns:
            # Drop rows with NaN in points column before filtering
            metadata = metadata[metadata[points_col].notna()]
            metadata = metadata[metadata[points_col] >= self.min_points]
            stats[f'Min {self.min_points} points at r{self.resolution}'] = len(metadata)
        
        return metadata, stats

    def _prefilter_edge_margin_instances(self) -> None:
        """
        One-time scan: drop instances whose sparse_index touches the boundary band.
        Faster than checking on every __getitem__ (avoids repeated npz IO + random retries).
        """
        n = len(self.instances)
        if n == 0:
            return

        workers = self.edge_scan_workers
        if workers is None:
            workers = min(32, max(1, (os.cpu_count() or 8)))

        try:
            from tqdm import tqdm as tqdm_bar
        except ImportError:
            def tqdm_bar(it, **kwargs):
                return it

        tasks = [
            (i, root, inst, self.resolution, self.edge_margin)
            for i, (root, inst) in enumerate(self.instances)
        ]
        keep_flags = [True] * n

        if workers <= 1:
            for t in tqdm_bar(tasks, total=n, desc="edge_margin filter", unit="npz"):
                i, keep, _err = _edge_margin_scan_one(t)
                keep_flags[i] = keep
        else:
            # map + chunksize：减少 submit 开销，结果顺序与 tasks 一致
            chunksize = max(1, min(128, n // (workers * 32)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = tqdm_bar(
                    ex.map(_edge_margin_scan_one, tasks, chunksize=chunksize),
                    total=n,
                    desc="edge_margin filter",
                    unit="npz",
                )
                for i, keep, _err in results:
                    keep_flags[i] = keep

        kept = [self.instances[i] for i in range(n) if keep_flags[i]]
        removed = n - len(kept)
        self.instances = kept
        self._edge_margin_prefiltered = True

        print(
            f"[SparseSDF] edge_margin={self.edge_margin} prefilter: kept {len(kept)}/{n}, "
            f"removed {removed} (mmap index-only scan, workers={workers})"
        )
        if len(kept) == 0:
            R = self.resolution
            m = self.edge_margin
            raise RuntimeError(
                "edge_margin 预过滤后没有剩余样本。\n"
                f"  当前 resolution={R}, edge_margin={m}：任一轴上存在坐标 <{m} 或 >={R - m} 即丢弃整例。\n"
                "  稀疏 SDF 往往铺满网格边界，margin 过大时会几乎全部被判触边。\n"
                "  处理：把 dataset.edge_margin 改小（例如 Stage2 且 latent 为 64 时常用 "
                f"{max(1, R // 64)}），或改为只裁剪坐标 / 换过滤策略。"
            )

    def _prefilter_encoder_band_instances(self) -> None:
        """
        一次性 mmap 扫描：丢弃在归一化 |SDF| 下没有任何「encoder 紧带」体素的 npz，
        避免训练时 N_enc=0（与 SparseSDF_VQVAETrainer 的 enc_mask 阈值一致）。
        """
        n = len(self.instances)
        if n == 0:
            return

        workers = self.encoder_band_scan_workers
        if workers is None:
            workers = self.edge_scan_workers
        if workers is None:
            workers = min(32, max(1, (os.cpu_count() or 8)))

        try:
            from tqdm import tqdm as tqdm_bar
        except ImportError:
            def tqdm_bar(it, **kwargs):
                return it

        thresh = self.encoder_input_sdf_thresh
        min_v = self.min_encoder_band_voxels
        tasks = [
            (i, root, inst, self.resolution, thresh, min_v)
            for i, (root, inst) in enumerate(self.instances)
        ]
        keep_flags = [True] * n

        if workers <= 1:
            for t in tqdm_bar(tasks, total=n, desc="encoder_band filter", unit="npz"):
                i, keep, _err = _encoder_band_scan_one(t)
                keep_flags[i] = keep
        else:
            chunksize = max(1, min(128, n // (workers * 32)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = tqdm_bar(
                    ex.map(_encoder_band_scan_one, tasks, chunksize=chunksize),
                    total=n,
                    desc="encoder_band filter",
                    unit="npz",
                )
                for i, keep, _err in results:
                    keep_flags[i] = keep

        kept = [self.instances[i] for i in range(n) if keep_flags[i]]
        removed = n - len(kept)
        self.instances = kept
        self._encoder_band_prefiltered = True

        print(
            f"[SparseSDF] encoder_band prefilter (|sdf|<={thresh:.4f}, min_count={min_v}): "
            f"kept {len(kept)}/{n}, removed {removed} (mmap SDF-only scan, workers={workers})"
        )
        if len(kept) == 0:
            raise RuntimeError(
                "encoder_band 预过滤后没有剩余样本。\n"
                f"  阈值 |sdf|<={thresh:.4f} 来自 input_band_factor/preprocessing_extra_band_factor。\n"
                "  请检查 npz 是否按相同 extra_band 归一化，或暂时放宽 min_encoder_band_voxels / 关闭 prefilter_encoder_band。"
            )

    def __getitem__(self, index):
        """
        在样本中附带 sha256 / 数据集下标，便于按训练步定位问题 npz（见 collate_fn）。
        """
        try:
            root, instance = self.instances[index]
            pack = self.get_instance(root, instance)
            pack['sha256'] = instance
            pack['dataset_index'] = int(index)
            pack['data_root'] = root
            return pack
        except Exception as e:
            print(e)
            return self.__getitem__(int(np.random.randint(0, len(self))))
    
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        """
        Load a single instance.
        
        Supports two directory layouts:
        1. Flat layout (sdf_voxelize.py output): {root}/{sha256}_r{resolution}.npz
        2. Subdirectory layout (legacy): {root}/sparse_sdf/{sha256}_r{resolution}.npz
        
        Args:
            root: Root directory
            instance: Instance SHA256 hash
        
        Returns:
            Dictionary containing:
                - sparse_sdf:   Tensor[N, 1]   signed SDF values
                - sparse_index: Tensor[N, 3]   grid coordinates
                - edge_mask:    BoolTensor[N]  True = sharp/edge voxel
        """
        global _EDGE_MASK_WARN_ISSUED

        # Load sparse SDF data - try flat layout first (sdf_voxelize.py output),
        # then fall back to subdirectory layout (compute_sparse_sdf.py output)
        sdf_path = os.path.join(root, f'{instance}_r{self.resolution}.npz')
        if not os.path.exists(sdf_path):
            sdf_path = os.path.join(root, 'sparse_sdf', f'{instance}_r{self.resolution}.npz')
        data = np.load(sdf_path)
        
        sparse_sdf = torch.from_numpy(data['sparse_sdf']).float()  # [N, 1]
        sparse_index = torch.from_numpy(data['sparse_index']).long()  # [N, 3]

        # Load edge_mask.  Older npz files don't have this key → fall back to
        # all-False (disables the sharp-region loss term for this sample) with a
        # one-time warning so the log isn't spammed.
        if 'edge_mask' in data:
            edge_mask = torch.from_numpy(data['edge_mask'].astype(np.bool_))  # [N]
        else:
            if not _EDGE_MASK_WARN_ISSUED:
                warnings.warn(
                    "[SparseSDF] 'edge_mask' not found in one or more npz files "
                    "(legacy data).  The sharp-region loss term will be disabled for "
                    "those samples.  Re-run sdf_voxelize.py with --extra_band_factor "
                    "to generate edge_mask.",
                    UserWarning,
                    stacklevel=2,
                )
                _EDGE_MASK_WARN_ISSUED = True
            edge_mask = torch.zeros(len(sparse_sdf), dtype=torch.bool)  # all-False fallback

        # Grid size for coordinate / GT lookup (must match sparse_index range)
        if 'resolution' in data:
            grid_resolution = int(np.asarray(data['resolution']).reshape(-1)[0])
        else:
            grid_resolution = int(self.resolution)

        if sparse_index.numel() > 0 and grid_resolution > 0:
            m = int(sparse_index.max().item())
            if m >= grid_resolution:
                raise ValueError(
                    f"[SparseSDF] {sdf_path}: sparse_index max={m} >= grid_resolution={grid_resolution}. "
                    "Corrupt npz or wrong dataset resolution."
                )

        # ── Encoder 紧带：全量 npz 上必须存在 |sdf|<=thresh 的体素（在 max_points 子采样之前检查）
        s_np = data["sparse_sdf"]
        abs_flat = np.abs(s_np.astype(np.float32, copy=False).reshape(-1))
        n_band = int(np.count_nonzero(abs_flat <= self.encoder_input_sdf_thresh))
        if n_band < self.min_encoder_band_voxels:
            raise ValueError(
                f"[SparseSDF] encoder_band_empty: {instance[:16]}.. |sdf|<={self.encoder_input_sdf_thresh:.4f} "
                f"count={n_band} < min_encoder_band_voxels={self.min_encoder_band_voxels} "
                f"(N={abs_flat.size}, min|sdf|={float(abs_flat.min()):.6f})."
            )

        # edge_margin: enforced once in __init__ via _prefilter_edge_margin_instances (fast path).

        # Subsample if too many points; keep edge_mask in sync
        if self.max_points is not None and len(sparse_sdf) > self.max_points:
            indices = torch.randperm(len(sparse_sdf))[:self.max_points]
            sparse_sdf = sparse_sdf[indices]
            sparse_index = sparse_index[indices]
            edge_mask = edge_mask[indices]
        
        return {
            'sparse_sdf':   sparse_sdf,
            'sparse_index': sparse_index,
            'edge_mask':    edge_mask,
            # Voxel grid size for this npz (matches filename …_r{R}.npz).  Trainer must
            # use this for GT lookup — NOT the VQVAE model's internal resolution.
            'grid_resolution': grid_resolution,
        }
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of instances.
        
        Args:
            batch: List of instances from get_instance
        
        Returns:
            Dictionary containing:
                - sparse_sdf:   Tensor[total_N, 1]
                - sparse_index: Tensor[total_N, 3]
                - edge_mask:    BoolTensor[total_N]  sharp-edge flags
                - batch_idx:    Tensor[total_N]       batch indices per point
                - grid_resolution: LongTensor scalar — voxel grid size R (same as npz)
        """
        # Concatenate all sparse data
        sparse_sdfs = []
        sparse_indices = []
        edge_masks = []
        batch_indices = []
        
        instance_ids = []
        dataset_indices = []
        data_roots = []
        grid_resolutions = []
        for batch_idx, item in enumerate(batch):
            n = len(item['sparse_sdf'])
            sparse_sdfs.append(item['sparse_sdf'])
            sparse_indices.append(item['sparse_index'])
            edge_masks.append(item['edge_mask'])
            batch_indices.append(
                torch.full((n,), batch_idx, dtype=torch.long)
            )
            instance_ids.append(item.get('sha256', ''))
            dataset_indices.append(int(item.get('dataset_index', -1)))
            data_roots.append(item.get('data_root', ''))
            grid_resolutions.append(int(item.get('grid_resolution', self.resolution)))

        if len(set(grid_resolutions)) != 1:
            raise ValueError(
                f"[SparseSDF.collate_fn] batch mixes grid_resolution values: {grid_resolutions}. "
                "Use a single dataset resolution per batch."
            )
        grid_resolution = grid_resolutions[0]

        return {
            'sparse_sdf':    torch.cat(sparse_sdfs,   dim=0),
            'sparse_index':  torch.cat(sparse_indices, dim=0),
            'edge_mask':     torch.cat(edge_masks,     dim=0),
            'batch_idx':     torch.cat(batch_indices,  dim=0),
            'grid_resolution': torch.tensor(grid_resolution, dtype=torch.long),
            'instance_ids':  instance_ids,
            'dataset_indices': torch.tensor(dataset_indices, dtype=torch.long),
            'data_roots':    data_roots,
        }

    def _grid_resolution_from_sample(self, sample: Dict[str, Any]) -> int:
        gr = sample.get('grid_resolution')
        if isinstance(gr, torch.Tensor) and gr.numel() > 0:
            return int(gr.detach().view(-1)[0].item())
        return int(self.resolution)

    @torch.no_grad()
    def _visualize_mc_mesh_renderer(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        from ..representations.mesh.cube2mesh import MeshExtractResult
        from ..renderers import MeshRenderer
        from ..utils.sparse_sdf_marching_cubes import sparse_sample_dict_to_trimeshes
        import utils3d

        sparse_sdf = sample['sparse_sdf']
        sparse_index = sample['sparse_index']
        batch_idx = sample['batch_idx']
        device = sparse_sdf.device
        if device.type != 'cuda':
            raise RuntimeError("MeshRenderer snapshot path requires CUDA")

        if batch_idx.numel() == 0:
            return torch.zeros(1, 3, 1024, 1024, device=device, dtype=torch.float32)

        vr = self._grid_resolution_from_sample(sample)
        meshes = sparse_sample_dict_to_trimeshes(sample, vr, self.snapshot_mc_threshold)
        batch_size = len(meshes)

        renderer_dev = (
            f'cuda:{device.index}' if device.index is not None else 'cuda'
        )
        renderer = MeshRenderer(
            rendering_options={'resolution': 512, 'near': 1.0, 'far': 100.0, 'ssaa': 2},
            device=renderer_dev,
        )

        yaws = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        pitch_list = [0.0, 0.0, 0.0, 0.0]
        fov = torch.deg2rad(torch.tensor(30.0, device=device))

        images: List[torch.Tensor] = []
        for b in range(min(batch_size, 16)):
            m = meshes[b] if b < len(meshes) else None
            if m is None or len(m.vertices) == 0:
                images.append(torch.zeros(3, 1024, 1024, device=device, dtype=torch.float32))
                continue
            verts = torch.from_numpy(np.asarray(m.vertices, dtype=np.float32)).to(device)
            faces = torch.from_numpy(np.asarray(m.faces, dtype=np.int64)).to(device)
            mer = MeshExtractResult(verts, faces)
            batch_images: List[torch.Tensor] = []
            for yaw, p in zip(yaws, pitch_list):
                orig = torch.tensor(
                    [
                        np.sin(yaw) * np.cos(p),
                        np.cos(yaw) * np.cos(p),
                        np.sin(p),
                    ],
                    dtype=torch.float32,
                    device=device,
                ) * 2.0
                target = torch.zeros(3, dtype=torch.float32, device=device)
                up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
                extrinsics = utils3d.torch.extrinsics_look_at(orig, target, up)
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                res = renderer.render(mer, extrinsics, intrinsics, return_types=['normal'])
                batch_images.append(res['normal'])
            top = torch.cat([batch_images[0], batch_images[1]], dim=2)
            bottom = torch.cat([batch_images[2], batch_images[3]], dim=2)
            grid = torch.cat([top, bottom], dim=1)
            images.append(grid)

        if len(images) == 0:
            return torch.zeros(1, 3, 1024, 1024, device=device, dtype=torch.float32)
        return torch.stack(images)

    @torch.no_grad()
    def _visualize_octree_voxels(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        from diffoctreerast import OctreeVoxelRasterizer  # noqa: F401
        from ..representations.octree import DfsOctree as Octree
        from ..renderers import OctreeRenderer
        import utils3d

        sparse_sdf = sample['sparse_sdf']
        sparse_index = sample['sparse_index']
        batch_idx = sample['batch_idx']
        device = sparse_sdf.device
        res_grid = self._grid_resolution_from_sample(sample)
        depth_bits = int(round(np.log2(res_grid))) if res_grid > 0 and (res_grid & (res_grid - 1)) == 0 else 10

        batch_size = int(batch_idx.max().item() + 1)
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 2
        renderer.pipe.primitive = 'voxel'

        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        pitch = [0, 0, 0, 0]
        images: List[torch.Tensor] = []

        for b in range(min(batch_size, 16)):
            mask = batch_idx == b
            coords = sparse_index[mask]
            if len(coords) == 0:
                # 与 MeshRenderer / matplotlib 退路一致：占位图，避免 torch.distributed.gather
                # 因各 rank 可视化张量 batch 维长度不一致而永久阻塞（NCCL 要求形状相同）。
                images.append(
                    torch.zeros(3, 1024, 1024, device=device, dtype=torch.float32)
                )
                continue

            dev = 'cuda' if coords.is_cuda else 'cpu'
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device=dev,
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            representation.position = coords.float() / float(res_grid)
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                depth_bits,
                dtype=torch.uint8,
                device=representation.position.device,
            )

            batch_images: List[torch.Tensor] = []
            for yaw, p in zip(yaws, pitch):
                orig = torch.tensor(
                    [
                        np.sin(yaw) * np.cos(p),
                        np.cos(yaw) * np.cos(p),
                        np.sin(p),
                    ],
                    dtype=torch.float32,
                    device=representation.position.device,
                ) * 2.0
                fov = torch.deg2rad(torch.tensor(30.0, device=representation.position.device))
                target = torch.zeros(3, dtype=torch.float32, device=representation.position.device)
                up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=representation.position.device)
                extrinsics = utils3d.torch.extrinsics_look_at(orig, target, up)
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                res = renderer.render(
                    representation,
                    extrinsics,
                    intrinsics,
                    colors_overwrite=representation.position,
                )
                batch_images.append(res['color'])

            top = torch.cat([batch_images[0], batch_images[1]], dim=2)
            bottom = torch.cat([batch_images[2], batch_images[3]], dim=2)
            grid = torch.cat([top, bottom], dim=1)
            images.append(grid)

        if len(images) == 0:
            return torch.zeros(1, 3, 1024, 1024, device=device, dtype=torch.float32)
        return torch.stack(images)

    @torch.no_grad()
    def _visualize_simple(
        self,
        sample: Dict[str, torch.Tensor],
        meshes: Optional[List[Any]] = None,
    ) -> torch.Tensor:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image

        sparse_sdf = sample['sparse_sdf']
        sparse_index = sample['sparse_index']
        batch_idx = sample['batch_idx']
        device = sparse_sdf.device

        if batch_idx.numel() == 0:
            return torch.zeros(1, 3, 1024, 1024, device=device, dtype=torch.float32)

        batch_size = int(batch_idx.max().item() + 1)
        if meshes is None:
            try:
                from ..utils.sparse_sdf_marching_cubes import sparse_sample_dict_to_trimeshes
                meshes = sparse_sample_dict_to_trimeshes(
                    sample,
                    self._grid_resolution_from_sample(sample),
                    self.snapshot_mc_threshold,
                )
            except Exception:
                meshes = [None] * batch_size

        per_view = 512
        dpi = 100
        fig_in = per_view / float(dpi)
        bg = (0, 0, 0)
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except AttributeError:
            resample = Image.LANCZOS  # type: ignore[attr-defined]

        images: List[torch.Tensor] = []
        for b in range(min(batch_size, 16)):
            mask = batch_idx == b
            pts = sparse_index[mask].detach().float().cpu().numpy()
            s = sparse_sdf[mask].detach().float().cpu().numpy().ravel()
            m = meshes[b] if b < len(meshes) else None

            if pts.shape[0] == 0:
                images.append(torch.zeros(3, 1024, 1024, device=device, dtype=torch.float32))
                continue

            parts: List[np.ndarray] = []
            if m is not None and len(m.vertices) > 0:
                vv = np.asarray(m.vertices, dtype=np.float64)
                ff = np.asarray(m.faces, dtype=np.int64)
                for azim in (0, 90, 180, 270):
                    fig = plt.figure(figsize=(fig_in, fig_in), dpi=dpi, facecolor=bg)
                    ax = fig.add_subplot(projection='3d')
                    ax.set_facecolor(bg)
                    ax.plot_trisurf(
                        vv[:, 0],
                        vv[:, 1],
                        vv[:, 2],
                        triangles=ff,
                        color='0.85',
                        linewidth=0.05,
                        antialiased=True,
                        shade=True,
                    )
                    ax.view_init(elev=20.0, azim=float(azim))
                    ax.set_axis_off()
                    lim = 1.05
                    ax.set_xlim(-lim, lim)
                    ax.set_ylim(-lim, lim)
                    ax.set_zlim(-lim, lim)
                    try:
                        ax.set_box_aspect((1, 1, 1))
                    except Exception:
                        pass
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    buf = BytesIO()
                    fig.savefig(buf, format='png', facecolor=bg, dpi=dpi, pad_inches=0)
                    plt.close(fig)
                    buf.seek(0)
                    parts.append(np.asarray(Image.open(buf).convert('RGB')))
            else:
                fig, axes = plt.subplots(2, 2, figsize=(2 * fig_in, 2 * fig_in), dpi=dpi, facecolor=bg)
                pairs = [(0, 1), (0, 2), (1, 2)]
                for ax, (i, j) in zip(axes.ravel()[:3], pairs):
                    ax.set_facecolor(bg)
                    ax.scatter(pts[:, i], pts[:, j], c=s, s=0.5, cmap='viridis', alpha=0.8)
                    ax.axis('off')
                axes.ravel()[3].set_facecolor(bg)
                axes.ravel()[3].scatter(pts[:, 0], pts[:, 1], c='w', s=0.3, alpha=0.25)
                axes.ravel()[3].axis('off')
                buf = BytesIO()
                fig.savefig(buf, format='png', facecolor=bg, dpi=dpi, pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                im = np.asarray(Image.open(buf).convert('RGB'))
                h, w = im.shape[:2]
                q = min(h, w) // 2
                parts = [
                    im[:q, :q],
                    im[:q, q:2 * q],
                    im[q:2 * q, :q],
                    im[q:2 * q, q:2 * q],
                ]

            row_t: List[torch.Tensor] = []
            for im in parts:
                pil = Image.fromarray(im)
                if im.shape[0] != per_view or im.shape[1] != per_view:
                    pil = pil.resize((per_view, per_view), resample)
                arr = np.asarray(pil.convert('RGB'))
                row_t.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
            top = torch.cat([row_t[0], row_t[1]], dim=2)
            bottom = torch.cat([row_t[2], row_t[3]], dim=2)
            grid = torch.cat([top, bottom], dim=1).to(device)
            images.append(grid)

        if len(images) == 0:
            return torch.zeros(1, 3, 1024, 1024, device=device, dtype=torch.float32)
        return torch.stack(images)

    @torch.no_grad()
    def visualize_sample(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Marching Cubes on sparse SDF (eval-aligned), then MeshRenderer 4-view 2×2 grid.

        Fallback order (白膜优先):
          1) CPU matplotlib + MC mesh（灰色 surf）
          2) Octree 体素伪彩色
          3) matplotlib 2D scatter（最后手段）

        各退路失败时会 ``warnings.warn``，避免静默降级误判为模型问题。
        """
        meshes_cache: Optional[List[Any]] = None
        try:
            return self._visualize_mc_mesh_renderer(sample)
        except Exception as e:
            warnings.warn(
                f"[SparseSDF.visualize_sample] CUDA MeshRenderer 路径失败: "
                f"{type(e).__name__}: {e}. 尝试 CPU matplotlib + Marching Cubes。",
                UserWarning,
                stacklevel=2,
            )

        try:
            from ..utils.sparse_sdf_marching_cubes import sparse_sample_dict_to_trimeshes

            meshes_cache = sparse_sample_dict_to_trimeshes(
                sample,
                self._grid_resolution_from_sample(sample),
                self.snapshot_mc_threshold,
            )
        except Exception as e:
            warnings.warn(
                f"[SparseSDF.visualize_sample] marching_cubes 失败: "
                f"{type(e).__name__}: {e}",
                UserWarning,
                stacklevel=2,
            )
            meshes_cache = None

        try:
            return self._visualize_simple(sample, meshes_cache)
        except Exception as e:
            warnings.warn(
                f"[SparseSDF.visualize_sample] matplotlib mesh 可视化失败: "
                f"{type(e).__name__}: {e}. 尝试 Octree。",
                UserWarning,
                stacklevel=2,
            )

        try:
            return self._visualize_octree_voxels(sample)
        except Exception as e:
            warnings.warn(
                f"[SparseSDF.visualize_sample] Octree 失败: "
                f"{type(e).__name__}: {e}. 使用 scatter 退路。",
                UserWarning,
                stacklevel=2,
            )
            return self._visualize_simple(sample, None)

