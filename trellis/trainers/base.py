from abc import abstractmethod
import os
import time
import json

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler


def convert_to_serializable(obj):
    """
    递归转换对象为 JSON 可序列化的类型
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    else:
        return obj


class Trainer:
    """
    Base class for training.

    Checkpoint retention: ``save_total_limit`` (default 3) caps how many distinct
    training-step checkpoint bundles are kept under ``output_dir/ckpts/`` after
    each ``save()``; set to 0 to disable pruning.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        max_epoch=None,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode='inflat_all',
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        disable_snapshot=False,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        save_total_limit=3,
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.max_epoch = max_epoch
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        self.disable_snapshot = disable_snapshot
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck
        # 磁盘：最多保留最近 N 套 checkpoint（与 i_save 对齐的 step 目录）；<=0 表示不限制
        self.save_total_limit = int(save_total_limit) if save_total_limit is not None else 3
        self._codebook_seen_codes_all = set()
        self._codebook_seen_codes_since_log = set()

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        
        # Calculate steps per epoch
        self.dataset_size = len(self.dataset)
        self.steps_per_epoch = int(np.ceil(self.dataset_size / self.batch_size))
        
        # Calculate max steps from max_epoch if specified
        if self.max_epoch is not None:
            max_steps_from_epoch = self.max_epoch * self.steps_per_epoch
            if self.max_steps is None:
                self.max_steps = max_steps_from_epoch
            else:
                # Use the minimum of max_steps and max_steps_from_epoch
                self.max_steps = min(self.max_steps, max_steps_from_epoch)
        elif self.max_steps is None:
            raise ValueError('Either max_steps or max_epoch must be specified.')
        
        # Load checkpoint
        self.step = 0
        self.epoch = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

        if self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)
            
    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        pass
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        sampler_seed = int(kwargs.get('seed', 0))
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
            seed=sampler_seed,
        )
        
        # 优化num_workers配置
        # 对于小数据集，减少worker数量以降低进程创建开销
        dataset_size = len(self.dataset)
        cpu_count = os.cpu_count()
        gpu_count = torch.cuda.device_count()
        
        if dataset_size < 50:
            # 小数据集：使用较少的worker
            num_workers = min(2, cpu_count // gpu_count)
        elif dataset_size < 200:
            # 中等数据集
            num_workers = min(4, cpu_count // gpu_count)
        else:
            # 大数据集：使用更多worker
            num_workers = min(8, int(np.ceil(cpu_count / gpu_count)))
        
        print(f"\n[DataLoader] Configuration:")
        print(f"  Dataset size: {dataset_size}")
        print(f"  Batch size per GPU: {self.batch_size_per_gpu}")
        print(f"  Num workers: {num_workers}")
        print(f"  Pin memory: True")
        print(f"  Persistent workers: {num_workers > 0}")
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
            prefetch_factor=2 if num_workers > 0 else None,  # 预取因子
        )
        self.data_iterator = cycle(self.dataloader)

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        pass
    
    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        pass
    
    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100):
        """
        Sample images from the dataset.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f'dataset_{k}', v) for k, v in vis.items()]
        else:
            save_cfg = [('dataset', vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        try:
            if self.is_master:
                print(f'\n📸 生成快照：{num_samples} 个样本...', end='', flush=True)

            if suffix is None:
                suffix = f'step{self.step:07d}'

            # Assign tasks
            num_samples_per_process = int(np.ceil(num_samples / self.world_size))
            
            samples = self.run_snapshot(
                num_samples_per_process,
                batch_size=batch_size,
                verbose=verbose,
                snapshot_export_suffix=suffix,
            )

            # Preprocess images
            for key in list(samples.keys()):
                if samples[key]['type'] == 'sample':
                    vis = self.visualize_sample(samples[key]['value'])
                    
                    if isinstance(vis, dict):
                        for k, v in vis.items():
                            samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                        del samples[key]
                    else:
                        samples[key] = {'value': vis, 'type': 'image'}
                        
        except Exception as e:
            print(f"\n❌ 快照生成错误:")
            print(f"  异常类型: {type(e).__name__}")
            print(f"  异常信息: {str(e)}")
            import traceback
            print(f"\n完整堆栈跟踪:")
            traceback.print_exc()
            raise

        # Gather results
        if self.world_size > 1:
            backend = dist.get_backend() if dist.is_initialized() else None
            exp_rows = int(np.ceil(num_samples / self.world_size))
            for key in samples.keys():
                if samples[key]['type'] != 'image':
                    continue
                value = samples[key]['value'].contiguous()
                # NCCL only supports CUDA tensors. Some visualize paths may return CPU tensors.
                if backend == 'nccl' and not value.is_cuda:
                    value = value.to(self.device, non_blocking=True)

                # NCCL gather 要求各 rank 的 tensor 形状与 rank0 的 gather 缓冲区一致。
                # 不同样本 / 不同可视化退路可能产生不同的 (B,H,W)，此处统一后再 gather。
                if isinstance(value, torch.Tensor) and value.ndim >= 3:
                    if value.ndim == 3:
                        value = value.unsqueeze(0)
                    h, w = int(value.shape[-2]), int(value.shape[-1])
                    hw = torch.tensor([h, w], device=value.device, dtype=torch.long)
                    dist.all_reduce(hw, op=dist.ReduceOp.MAX)
                    h_max, w_max = int(hw[0].item()), int(hw[1].item())
                    if h != h_max or w != w_max:
                        vf = value.float()
                        value = F.interpolate(
                            vf, size=(h_max, w_max), mode='bilinear', align_corners=False
                        ).to(dtype=value.dtype)
                    b = int(value.shape[0])
                    if b > exp_rows:
                        value = value[:exp_rows]
                    elif b < exp_rows:
                        pad = torch.zeros(
                            exp_rows - b,
                            *value.shape[1:],
                            device=value.device,
                            dtype=value.dtype,
                        )
                        value = torch.cat([value, pad], dim=0)
                    value = value.contiguous()

                if self.is_master:
                    all_images = [torch.empty_like(value) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(value, all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    continue
                value = samples[key]['value'].contiguous()
                if backend == 'nccl' and not value.is_cuda:
                    value = value.to(self.device, non_blocking=True)
                if self.is_master:
                    all_images = [torch.empty_like(value) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(value, all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    utils.save_image(
                        samples[key]['value'],
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]['type'] == 'number':
                    min = samples[key]['value'].min()
                    max = samples[key]['value'].max()
                    images = (samples[key]['value'] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min}, max: {max}',
                    )

        if self.is_master:
            print(' ✅ 完成')

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        pass

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass
    
    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                # 第一次预取
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            # 预取下一批
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a training step.
        """
        pass

    def run(self):
        """
        Run training.
        """
        if self.is_master:
            print('\n' + '='*100)
            print('🚀 开始训练...')
            print('='*100)
            if not self.disable_snapshot:
                self.snapshot_dataset()
            else:
                print('⚠️  快照功能已被配置禁用')
        
        if not self.disable_snapshot:
            if self.step == 0:
                self.snapshot(suffix='init')
            else: # resume
                self.snapshot(suffix=f'resume_step{self.step:07d}')
        else:
            if self.is_master:
                print('⚠️  跳过初始快照（已被配置禁用）')

        log = []
        time_last_print = 0.0
        time_elapsed = 0.0
        time_data_load = 0.0
        time_forward = 0.0
        time_backward = 0.0

        def _normalize_codebook_unique_codes(raw_codes):
            if raw_codes is None:
                return []
            if isinstance(raw_codes, torch.Tensor):
                raw_codes = raw_codes.detach().cpu().reshape(-1).tolist()
            elif isinstance(raw_codes, np.ndarray):
                raw_codes = raw_codes.reshape(-1).tolist()
            elif isinstance(raw_codes, (list, tuple, set)):
                raw_codes = list(raw_codes)
            else:
                raw_codes = [raw_codes]

            normalized_codes = []
            for code in raw_codes:
                try:
                    normalized_codes.append(int(code))
                except (TypeError, ValueError):
                    continue
            return normalized_codes
        
        if self.is_master:
            print(f"\n{'='*100}")
            print(f"📊 训练循环开始")
            print(f"{'='*100}")
            print(f"\n📦 数据集信息:")
            print(f"  数据集大小: {self.dataset_size:,} 样本")
            print(f"  批大小: {self.batch_size} (每GPU: {self.batch_size_per_gpu}, 世界大小: {self.world_size})")
            print(f"  每个 Epoch 步数: {self.steps_per_epoch:,} steps")
            
            print(f"\n🎯 训练目标:")
            print(f"  最大步数: {self.max_steps:,} steps")
            if self.max_epoch is not None:
                print(f"  最大 Epoch: {self.max_epoch}")
                total_epochs = self.max_steps / self.steps_per_epoch
                print(f"  将训练约 {total_epochs:.2f} 个 epoch")
            else:
                total_epochs = self.max_steps / self.steps_per_epoch
                print(f"  将训练约 {total_epochs:.2f} 个 epoch")
            print(f"  总计将处理约 {self.max_steps * self.batch_size:,} 个样本次数")
            
            print(f"\n🔄 训练状态:")
            print(f"  起始步数: {self.step:,}")
            print(f"  起始 Epoch: {self.epoch}")
            print(f"  剩余步数: {self.max_steps - self.step:,}")
            print(f"{'='*100}\n")
        
        while self.step < self.max_steps:
            step_start_time = time.time()
            
            # 数据加载
            data_load_start = time.time()
            data_list = self.load_data()
            data_load_time = time.time() - data_load_start
            time_data_load += data_load_time
            
            # 训练步骤
            forward_backward_start = time.time()
            step_log = self.run_step(data_list)
            forward_backward_time = time.time() - forward_backward_start
            
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            time_elapsed += step_time

            self.step += 1
            if self.is_master and step_log is not None and 'status' in step_log:
                status_info = step_log['status']
                if 'codebook_batch_unique_codes' in status_info:
                    batch_unique_codes = _normalize_codebook_unique_codes(
                        status_info.pop('codebook_batch_unique_codes')
                    )
                    self._codebook_seen_codes_all.update(batch_unique_codes)
                    self._codebook_seen_codes_since_log.update(batch_unique_codes)
            
            # Update epoch counter
            self.epoch = self.step // self.steps_per_epoch
            steps_in_epoch = self.step % self.steps_per_epoch

            # 详细的步骤级打印（每个step都打印关键信息）
            if self.is_master:
                # 提取loss信息
                if step_log is not None and 'loss' in step_log:
                    loss_info = step_log['loss']
                    loss_str = f"Loss: {loss_info.get('loss', 0.0):.6f}"
                    if 'recon' in loss_info:
                        loss_str += f" (Recon: {loss_info['recon']:.6f}"
                    if 'vq' in loss_info:
                        loss_str += f", VQ: {loss_info['vq']:.6f}"
                    if 'commitment' in loss_info:
                        loss_str += f", Commit: {loss_info['commitment']:.6f})"
                    elif 'recon' in loss_info:
                        loss_str += ")"
                else:
                    loss_str = "Loss: N/A"
                
                # 每个step打印简要信息
                print(f"[Epoch {self.epoch} | Step {self.step:>6}/{self.max_steps} ({steps_in_epoch}/{self.steps_per_epoch})] {loss_str} | "
                      f"时间: {step_time:.3f}s (数据: {data_load_time:.3f}s, 训练: {forward_backward_time:.3f}s)", 
                      flush=True)
            
            # Print progress summary（每i_print步打印详细统计）
            if self.is_master and self.step % self.i_print == 0:
                time_interval = time_elapsed - time_last_print
                speed = self.i_print / time_interval * 3600 if time_interval > 0 else 0
                avg_data_time = time_data_load / self.i_print
                avg_train_time = (time_elapsed - time_last_print - time_data_load) / self.i_print
                
                print(f"\n{'='*100}")
                print(f"📈 训练进度汇总 [Epoch {self.epoch} | Step {self.step:,}]")
                print(f"{'='*100}")
                print(f"  Step 进度: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)")
                print(f"  Epoch 进度: {self.epoch}.{steps_in_epoch} / {self.max_steps / self.steps_per_epoch:.2f} epochs")
                print(f"  已处理样本次数: {self.step * self.batch_size:,}")
                print(f"  已用时间: {time_elapsed / 3600:.2f} 小时")
                print(f"  训练速度: {speed:.2f} steps/小时")
                print(f"  预计剩余: {(self.max_steps - self.step) / speed:.2f} 小时" if speed > 0 else "  预计剩余: 计算中...")
                print(f"  平均数据加载: {avg_data_time:.3f}秒/step")
                print(f"  平均训练时间: {avg_train_time:.3f}秒/step")
                print(f"  数据加载占比: {avg_data_time / (avg_data_time + avg_train_time) * 100:.1f}%")
                
                if step_log is not None and 'status' in step_log:
                    status_info = step_log['status']
                    if 'lr' in status_info:
                        print(f"  当前学习率: {status_info['lr']:.2e}")
                    if 'grad_norm' in status_info:
                        print(f"  梯度范数: {status_info['grad_norm']:.6f}")
                    
                    # 码本利用率统计
                    if 'codebook_perplexity' in status_info:
                        # 获取码本总数
                        try:
                            # 尝试从模型中获取码本大小
                            if 'vqvae' in self.models:
                                vqvae_model = self.models['vqvae']
                                if hasattr(vqvae_model, 'module'):
                                    vqvae_model = vqvae_model.module
                                if hasattr(vqvae_model, 'vq') and hasattr(vqvae_model.vq, 'num_embeddings'):
                                    codebook_size = vqvae_model.vq.num_embeddings
                                else:
                                    codebook_size = None
                            else:
                                codebook_size = None
                        except:
                            codebook_size = None
                        
                        print(f"\n📊 码本利用率统计 (per-sample 均值):")
                        if codebook_size is not None:
                            print(f"  - 困惑度 (Perplexity): {status_info['codebook_perplexity']:.2f} / {codebook_size}")
                        else:
                            print(f"  - 困惑度 (Perplexity): {status_info['codebook_perplexity']:.2f}")
                        print(f"  - 信息熵 (Entropy): {status_info['codebook_entropy']:.4f}")
                        print(f"  - 活跃码本比例 (per-sample): {status_info['codebook_utilization_ratio']:.2f}%")
                        if codebook_size is not None and codebook_size > 0:
                            all_sample_active_ratio = (
                                100.0 * len(self._codebook_seen_codes_all) / float(codebook_size)
                            )
                            i_log_window_active_ratio = (
                                100.0 * len(self._codebook_seen_codes_since_log) / float(codebook_size)
                            )
                            print(f"  - 活跃码本比例 (all sample 累计): {all_sample_active_ratio:.2f}%")
                            print(f"  - 活跃码本比例 (上一i_log到当前累计): {i_log_window_active_ratio:.2f}%")
                        if codebook_size is not None:
                            print(f"  - 唯一码本数量 (per-sample): {status_info['codebook_unique_count']:.1f} / {codebook_size}")
                            if 'codebook_batch_unique_count' in status_info:
                                print(f"  - 唯一码本数量 (batch整体): {status_info['codebook_batch_unique_count']:.0f} / {codebook_size}")
                        else:
                            print(f"  - 唯一码本数量 (per-sample): {status_info['codebook_unique_count']:.1f}")
                            if 'codebook_batch_unique_count' in status_info:
                                print(f"  - 唯一码本数量 (batch整体): {status_info['codebook_batch_unique_count']:.0f}")
                
                print(f"{'='*100}\n")
                
                time_last_print = time_elapsed
                time_data_load = 0.0  # 重置计数器

            # Check ddp
            if self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # Sample images
            if self.step % self.i_sample == 0 and not self.disable_snapshot:
                self.snapshot()

            if self.is_master:
                log.append((self.step, {}))

                # Log time
                log[-1][1]['time'] = {
                    'step': step_time,
                    'elapsed': time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    log[-1][1].update(step_log)

                # Log scale
                if self.fp16_mode == 'amp':
                    log[-1][1]['scale'] = self.scaler.get_scale()
                elif self.fp16_mode == 'inflat_all':
                    log[-1][1]['log_scale'] = self.log_scale

                # Save log
                if self.step % self.i_log == 0:
                    ## save to log file
                    log_str = '\n'.join([
                        f'{step}: {json.dumps(convert_to_serializable(log))}' for step, log in log
                    ])
                    with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
                        log_file.write(log_str + '\n')

                    # show with mlflow
                    log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
                    log_show = dict_reduce(log_show, lambda x: np.mean(x))
                    log_show = dict_flatten(log_show, sep='/')
                    for key, value in log_show.items():
                        self.writer.add_scalar(key, value, self.step)
                    log = []
                    self._codebook_seen_codes_since_log.clear()

                # Save checkpoint
                if self.step % self.i_save == 0:
                    self.save()

        if self.is_master:
            self.snapshot(suffix='final')
            self.writer.close()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
            