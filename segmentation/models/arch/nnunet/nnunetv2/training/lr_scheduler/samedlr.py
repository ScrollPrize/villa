import torch
from torch.optim.lr_scheduler import _LRScheduler

# Custom LR Scheduler Implementation
class CustomWarmupDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_period, max_iterations, base_lr, weight_decay, last_epoch=-1, verbose=False):
        self.warmup_period = warmup_period
        self.max_iterations = max_iterations
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_period:
            return [self.base_lr * ((self.last_epoch + 1) / self.warmup_period) for _ in self.optimizer.param_groups]
        else:
            if self.warmup_period:
                shift_iter = self.last_epoch - self.warmup_period
            else:
                shift_iter = self.last_epoch
            return [self.base_lr * (1.0 - shift_iter / self.max_iterations) ** self.weight_decay for _ in self.optimizer.param_groups]

