from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmUpScheduler(_LRScheduler):
    """Linearly increases learning rate from 0 to base_lr over `warmup_steps` steps.
    After warmup_steps, returns to using the original scheduler.
    """
    def __init__(self, optimizer, warmup_steps, base_scheduler=None, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)
        # Initialize _last_lr for get_last_lr() support
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / float(self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        if self.base_scheduler is not None:
            try:
                return self.base_scheduler.get_lr()
            except (AttributeError, NotImplementedError):
                return [group['lr'] for group in self.optimizer.param_groups]
        return self.base_lrs

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / float(self.warmup_steps)
            values = [base_lr * alpha for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group['lr'] = lr
            self._last_lr = values
        elif self.base_scheduler is not None:
            self.base_scheduler.step(self.last_epoch - self.warmup_steps)
            try:
                self._last_lr = self.base_scheduler.get_last_lr()
            except (AttributeError, NotImplementedError):
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        else:
            self._last_lr = self.base_lrs
            for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
                param_group['lr'] = lr
