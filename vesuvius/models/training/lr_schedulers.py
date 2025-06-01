"""
Learning rate schedulers for training neural networks.

This module contains various learning rate scheduling strategies including
the PolyLR scheduler used by nnU-Net.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional

from pytorch_optimizer import CosineAnnealingWarmupRestarts


class PolyLRScheduler(_LRScheduler):
    """
    Polynomial learning rate scheduler as used in nnU-Net.
    
    The learning rate is decayed using a polynomial function:
    lr = initial_lr * (1 - current_step / max_steps) ** exponent
    
    This provides a smooth decay that starts fast and slows down towards the end,
    which has been shown to work well for medical image segmentation tasks.
    
    Args:
        optimizer: The optimizer to schedule
        initial_lr: The initial learning rate
        max_steps: The maximum number of steps (typically num_epochs)
        exponent: The polynomial exponent (default: 0.9)
        current_step: The current step to start from (default: None)
    
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> scheduler = PolyLRScheduler(optimizer, initial_lr=0.01, max_steps=100)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step(epoch)
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer, 
                 initial_lr: float, 
                 max_steps: int, 
                 exponent: float = 0.9, 
                 current_step: Optional[int] = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        # In newer PyTorch versions, _LRScheduler only takes optimizer and last_epoch
        # The verbose parameter was removed
        super().__init__(optimizer, last_epoch=current_step if current_step is not None else -1)

    def step(self, current_step: Optional[int] = None):
        """
        Update the learning rate.
        
        Args:
            current_step: The current step. If None, uses internal counter.
        """
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


# Additional common schedulers that might be useful

class WarmupPolyLRScheduler(_LRScheduler):
    """
    Polynomial learning rate scheduler with linear warmup.
    
    Combines a linear warmup phase with polynomial decay. During warmup,
    the learning rate increases linearly from 0 to initial_lr.
    
    Args:
        optimizer: The optimizer to schedule
        initial_lr: The initial learning rate (reached after warmup)
        max_steps: The maximum number of steps
        warmup_steps: Number of warmup steps
        exponent: The polynomial exponent (default: 0.9)
        current_step: The current step to start from (default: None)
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 initial_lr: float,
                 max_steps: int,
                 warmup_steps: int,
                 exponent: float = 0.9,
                 current_step: Optional[int] = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.exponent = exponent
        self.ctr = 0
        # In newer PyTorch versions, _LRScheduler only takes optimizer and last_epoch
        # The verbose parameter was removed
        super().__init__(optimizer, last_epoch=current_step if current_step is not None else -1)
    
    def step(self, current_step: Optional[int] = None):
        """
        Update the learning rate with warmup support.
        
        Args:
            current_step: The current step. If None, uses internal counter.
        """
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        
        if current_step < self.warmup_steps:
            # Linear warmup
            new_lr = self.initial_lr * (current_step / self.warmup_steps)
        else:
            # Polynomial decay after warmup
            steps_after_warmup = current_step - self.warmup_steps
            max_steps_after_warmup = self.max_steps - self.warmup_steps
            new_lr = self.initial_lr * (1 - steps_after_warmup / max_steps_after_warmup) ** self.exponent
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def get_scheduler(scheduler_type: str, 
                  optimizer: torch.optim.Optimizer,
                  initial_lr: float,
                  max_steps: int,
                  **kwargs) -> _LRScheduler:
    """
    Factory function to create learning rate schedulers.
    
    Args:
        scheduler_type: Type of scheduler ('poly', 'warmup_poly', 'cosine', 'cosine_warmup', 'step')
        optimizer: The optimizer to schedule
        initial_lr: The initial learning rate
        max_steps: The maximum number of steps
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        The configured learning rate scheduler
        
    Example:
        >>> scheduler = get_scheduler('poly', optimizer, 0.01, 100, exponent=0.9)
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'poly':
        exponent = kwargs.get('exponent', 0.9)
        return PolyLRScheduler(optimizer, initial_lr, max_steps, exponent)
    
    elif scheduler_type == 'warmup_poly':
        warmup_steps = kwargs.get('warmup_steps', int(0.1 * max_steps))
        exponent = kwargs.get('exponent', 0.9)
        return WarmupPolyLRScheduler(optimizer, initial_lr, max_steps, warmup_steps, exponent)
    
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    
    elif scheduler_type == 'cosine_warmup':
        # Extract parameters for cosine warmup scheduler
        first_cycle_steps = kwargs.get('first_cycle_steps', max_steps)
        cycle_mult = kwargs.get('cycle_mult', 1.0)
        warmup_steps = kwargs.get('warmup_steps', int(0.1 * first_cycle_steps))  # Default 10% warmup
        gamma = kwargs.get('gamma', 0.9)
        min_lr = kwargs.get('min_lr', 1e-6)
        
        # CosineAnnealingWarmupRestarts expects max_lr not initial_lr
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=cycle_mult,
            max_lr=initial_lr,  # Use initial_lr as max_lr
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            gamma=gamma
        )
    
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', max_steps // 3)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
