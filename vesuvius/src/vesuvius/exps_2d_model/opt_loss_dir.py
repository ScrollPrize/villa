from __future__ import annotations

import torch

import fit_data


def direction_loss_map(*, model, data: fit_data.FitData) -> tuple[torch.Tensor, torch.Tensor]:
	data_s, mask = data.grid_sample(model=model)
	dir0_pred, dir1_pred = model.direction_encoding(shape=data_s.dir0.shape)
	diff0 = dir0_pred - data_s.dir0
	diff1 = dir1_pred - data_s.dir1
	lm = 0.5 * (diff0 * diff0 + diff1 * diff1)
	return lm * mask, mask


def direction_loss(*, model, data: fit_data.FitData) -> torch.Tensor:
	"""Direction-only loss vs UNet-style (dir0, dir1) encodings."""
	lm, mask = direction_loss_map(model=model, data=data)
	wsum = mask.sum()
	if float(wsum.detach().cpu()) > 0.0:
		return lm.sum() / wsum
	return lm.mean()
