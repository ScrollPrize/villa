import torch
from torch.optim import AdamW, SGD

from train_resnet3d_lib.config import CFG
from warmup_scheduler import GradualWarmupScheduler


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """Compatibility wrapper used by older training configs."""

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler
        )

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0) for base_lr in self.base_lrs]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 50, eta_min=1e-6
    )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine
    )
    return scheduler


def configure_optimizers(model):
    lr = float(CFG.lr)
    weight_decay = float(getattr(CFG, "weight_decay", 0.0) or 0.0)
    optimizer_name = str(getattr(CFG, "optimizer", "adamw")).strip().lower()

    exclude_wd_bias_norm = bool(getattr(CFG, "exclude_weight_decay_bias_norm", False))
    use_param_groups = exclude_wd_bias_norm and weight_decay > 0
    decay_params = []
    no_decay_params = []
    if use_param_groups:
        for _, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if int(getattr(param, "ndim", 0)) < 2:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    if optimizer_name == "adamw":
        beta2 = float(getattr(CFG, "adamw_beta2", 0.999))
        eps = float(getattr(CFG, "adamw_eps", 1e-8))
        betas = (0.9, beta2)
        if use_param_groups:
            optimizer = AdamW(
                [
                    {"params": decay_params, "weight_decay": weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=0.0,
            )
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
    elif optimizer_name == "sgd":
        momentum = float(getattr(CFG, "sgd_momentum", 0.0) or 0.0)
        nesterov = bool(getattr(CFG, "sgd_nesterov", False))
        if use_param_groups:
            optimizer = SGD(
                [
                    {"params": decay_params, "weight_decay": weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=0.0,
            )
        else:
            optimizer = SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
            )
    else:
        raise ValueError(
            f"Unsupported optimizer={CFG.optimizer!r}. Supported: 'adamw' | 'sgd'."
        )

    scheduler_name = str(getattr(CFG, "scheduler", "OneCycleLR")).lower()
    steps_per_epoch = int(model.total_steps)
    epochs = int(CFG.epochs)

    if scheduler_name == "onecyclelr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CFG.lr,
            pct_start=float(getattr(CFG, "onecycle_pct_start", 0.15)),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            div_factor=float(getattr(CFG, "onecycle_div_factor", 25.0)),
            final_div_factor=float(getattr(CFG, "onecycle_final_div_factor", 1e2)),
        )
        interval = "step"
    elif scheduler_name == "cosine":
        total_steps = max(1, steps_per_epoch * epochs)
        warmup_pct = float(getattr(CFG, "cosine_warmup_pct", 0.0) or 0.0)
        warmup_pct = max(0.0, min(1.0, warmup_pct))
        warmup_steps = int(round(total_steps * warmup_pct))
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

        eta_min = float(getattr(CFG, "min_lr", 0.0))

        if warmup_steps > 0:
            warmup_factor = float(getattr(CFG, "warmup_factor", 1.0) or 1.0)
            if warmup_factor <= 0:
                raise ValueError(f"warmup_factor must be > 0, got {warmup_factor}")
            start_factor = 1.0 / warmup_factor

            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=float(start_factor),
                end_factor=1.0,
                total_iters=int(warmup_steps),
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(total_steps - warmup_steps),
                eta_min=float(eta_min),
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[int(warmup_steps)],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(total_steps),
                eta_min=float(eta_min),
            )
        interval = "step"
    elif scheduler_name == "gradualwarmupschedulerv2":
        scheduler = get_scheduler(CFG, optimizer)
        interval = "epoch"
    else:
        raise ValueError(
            f"Unsupported scheduler={CFG.scheduler!r}. Supported: 'OneCycleLR' | 'cosine' | 'GradualWarmupSchedulerV2'."
        )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": interval,
        },
    }


__all__ = [
    "GradualWarmupSchedulerV2",
    "get_scheduler",
    "configure_optimizers",
]
