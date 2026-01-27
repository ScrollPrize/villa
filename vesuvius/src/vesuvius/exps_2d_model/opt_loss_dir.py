from __future__ import annotations

import torch

import fit_data
import model as fit_model


def direction_loss_map(*, res: fit_model.FitResult) -> tuple[torch.Tensor, torch.Tensor]:
	diff0 = res.dir0_pred - res.data_s.dir0
	diff1 = res.dir1_pred - res.data_s.dir1
	lm = 0.5 * (diff0 * diff0 + diff1 * diff1)
	return lm * res.mask, res.mask


def direction_loss(*, res: fit_model.FitResult) -> torch.Tensor:
	"""Direction-only loss vs UNet-style (dir0, dir1) encodings."""
	lm, _mask = direction_loss_map(res=res)
	wsum = res.mask.sum()
	if float(wsum.detach().cpu()) > 0.0:
		return lm.sum() / wsum
	return lm.mean()
