from __future__ import annotations

import torch

import model as fit_model


def mod_smooth_y_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return (lm, mask) for y-smoothness of modulation params (amp & bias) on the base mesh."""
	amp = res.amp_lr
	bias = res.bias_lr
	mods = torch.cat([amp, bias], dim=1)
	if int(mods.shape[2]) < 2:
		base = torch.zeros((), device=mods.device, dtype=mods.dtype)
		mask0 = torch.zeros((), device=mods.device, dtype=mods.dtype)
		return base, mask0

	dy = mods[:, :, 1:, :] - mods[:, :, :-1, :]
	lm = (dy * dy).mean(dim=1, keepdim=True)
	mask_lr = res.mask_lr
	mask = torch.minimum(mask_lr[:, :, 1:, :], mask_lr[:, :, :-1, :])
	return lm, mask


def _masked_mean(lm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	wsum = mask.sum()
	if float(wsum) > 0.0:
		return (lm * mask).sum() / wsum
	return lm.mean()


def mod_smooth_y_loss(*, res: fit_model.FitResult) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	lm, mask = mod_smooth_y_loss_map(res=res)
	return _masked_mean(lm, mask), (lm,), (mask,)


def contr_loss(*, res: fit_model.FitResult) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	amp = res.amp_lr
	bias = res.bias_lr
	loss = ((amp - 1.0) * (amp - 1.0) + bias * bias).mean()
	return loss, tuple(), tuple()
