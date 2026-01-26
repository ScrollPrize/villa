from __future__ import annotations

import torch

import fit_data


def direction_loss(*, model, data: fit_data.FitData) -> torch.Tensor:
	"""Direction-only loss vs UNet-style (dir0, dir1) encodings."""
	dir0_pred, dir1_pred = model.direction_encoding(shape=data.dir0.shape)
	diff0 = dir0_pred - data.dir0
	diff1 = dir1_pred - data.dir1
	return 0.5 * ((diff0 * diff0).mean() + (diff1 * diff1).mean())

