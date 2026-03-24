from __future__ import annotations

from dataclasses import dataclass

from torch.optim import AdamW as TorchAdamW
from torch.optim import SGD as TorchSGD


def _build_parameter_groups(model, *, weight_decay: float, exclude_weight_decay_bias_norm: bool):
    """Split parameters so bias/norm-like tensors can skip weight decay when requested."""
    weight_decay = float(weight_decay)
    if not exclude_weight_decay_bias_norm or weight_decay <= 0.0:
        return model.parameters(), weight_decay

    decay_params = []
    no_decay_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if int(getattr(param, "ndim", 0)) < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], 0.0


@dataclass(frozen=True)
class AdamWOptimizer:
    lr: float = 2e-5
    weight_decay: float = 1e-6
    beta2: float = 0.999
    eps: float = 1e-8
    exclude_weight_decay_bias_norm: bool = True

    def build(self, model):
        params, optimizer_weight_decay = _build_parameter_groups(
            model,
            weight_decay=float(self.weight_decay),
            exclude_weight_decay_bias_norm=bool(self.exclude_weight_decay_bias_norm),
        )
        return TorchAdamW(
            params,
            lr=float(self.lr),
            betas=(0.9, float(self.beta2)),
            eps=float(self.eps),
            weight_decay=optimizer_weight_decay,
        )


@dataclass(frozen=True)
class SGDOptimizer:
    lr: float = 2e-5
    weight_decay: float = 1e-6
    momentum: float = 0.9
    nesterov: bool = False
    exclude_weight_decay_bias_norm: bool = True

    def build(self, model):
        params, optimizer_weight_decay = _build_parameter_groups(
            model,
            weight_decay=float(self.weight_decay),
            exclude_weight_decay_bias_norm=bool(self.exclude_weight_decay_bias_norm),
        )
        return TorchSGD(
            params,
            lr=float(self.lr),
            momentum=float(self.momentum),
            nesterov=bool(self.nesterov),
            weight_decay=optimizer_weight_decay,
        )


@dataclass(frozen=True)
class MuonOptimizer:
    lr: float = 2e-5
    weight_decay: float = 1e-6
    momentum: float = 0.95
    nesterov: bool = False
    ns_steps: int = 5
    adjust_lr_fn: str | None = "match_rms_adamw"
    adamw_lr: float | None = None
    beta2: float = 0.95
    eps: float = 1e-8
    conv_mode: str = "flatten"
    normalize_spatial: bool = True

    def build(self, model):
        try:
            from timm.optim import Muon as TimmMuon
        except ImportError as exc:
            raise ImportError("MuonOptimizer requires timm.optim.Muon to be installed") from exc

        return TimmMuon(
            model.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
            momentum=float(self.momentum),
            nesterov=bool(self.nesterov),
            ns_steps=int(self.ns_steps),
            adjust_lr_fn=self.adjust_lr_fn,
            adamw_lr=None if self.adamw_lr is None else float(self.adamw_lr),
            betas=(0.9, float(self.beta2)),
            eps=float(self.eps),
            conv_mode=str(self.conv_mode),
            normalize_spatial=bool(self.normalize_spatial),
        )
