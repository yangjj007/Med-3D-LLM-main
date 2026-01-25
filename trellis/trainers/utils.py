import torch.nn as nn


# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    """
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )
    

def zero_grad(model_params):
    for param in model_params:
       if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()
            

# LR Schedulers
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

class LinearWarmupLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0


class WarmupThenScheduler(_LRScheduler):
    """
    带有 Warmup 阶段的学习率调度器
    
    在前 warmup_steps 步内，学习率线性增长从 0 到 base_lr
    Warmup 结束后，使用指定的主调度器（如 CosineAnnealingLR）
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: warmup 步数
        main_scheduler_class: warmup 后使用的调度器类（如 torch.optim.lr_scheduler.CosineAnnealingLR）
        main_scheduler_kwargs: 主调度器的参数字典
        last_epoch: 上次的 epoch 编号（用于从 checkpoint 恢复）
    
    Example:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        scheduler = WarmupThenScheduler(
            optimizer,
            warmup_steps=500,
            main_scheduler_class=CosineAnnealingLR,
            main_scheduler_kwargs={'T_max': 100000, 'eta_min': 5e-5}
        )
    """
    def __init__(self, optimizer, warmup_steps, main_scheduler_class, main_scheduler_kwargs, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.main_scheduler_class = main_scheduler_class
        self.main_scheduler_kwargs = main_scheduler_kwargs
        
        # 创建主调度器（warmup 后使用）
        self.main_scheduler = main_scheduler_class(optimizer, **main_scheduler_kwargs)
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super(WarmupThenScheduler, self).__init__(optimizer, last_epoch)
        
        # 初始化 _last_lr（用于 get_last_lr()）
        self._last_lr = [group['lr'] for group in optimizer.param_groups]
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup 阶段：线性增长从 0 到 base_lr
            warmup_factor = float(self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Warmup 后：使用主调度器
            # 注意：需要调整主调度器的 last_epoch，使其从 warmup 结束后开始计数
            adjusted_epoch = self.last_epoch - self.warmup_steps
            self.main_scheduler.last_epoch = adjusted_epoch
            return self.main_scheduler.get_lr()
    
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        # 计算新的学习率
        values = self.get_lr()
        
        # 更新优化器的学习率
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        
        # 保存当前学习率（用于 get_last_lr()）
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        # 如果已经过了 warmup 阶段，同时更新主调度器
        if self.last_epoch >= self.warmup_steps:
            adjusted_epoch = self.last_epoch - self.warmup_steps
            self.main_scheduler.last_epoch = adjusted_epoch
    
    def state_dict(self):
        """返回调度器的状态字典"""
        state = {
            'warmup_steps': self.warmup_steps,
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs,
            '_last_lr': self._last_lr,
            'main_scheduler': self.main_scheduler.state_dict()
        }
        return state
    
    def load_state_dict(self, state_dict):
        """从状态字典加载调度器"""
        self.warmup_steps = state_dict['warmup_steps']
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']
        self._last_lr = state_dict.get('_last_lr', self.base_lrs)  # 兼容旧版本
        self.main_scheduler.load_state_dict(state_dict['main_scheduler'])
        