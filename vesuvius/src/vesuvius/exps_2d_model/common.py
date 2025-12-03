from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
	"""Two conv-bn-relu blocks used in the UNet encoder & decoder."""

	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class UNet(nn.Module):
	"""Configurable 2D U-Net for 1-channel image input (+2 coord channels) & N outputs."""

	def __init__(
		self,
		in_channels: int = 1,
		out_channels: int = 3,
		base_channels: int = 32,
		num_levels: int = 9,
		max_channels: int = 256,
	) -> None:
		super().__init__()
		if num_levels < 2:
			raise ValueError(f"num_levels must be >= 2, got {num_levels}")
		self.num_levels = num_levels
		self.max_channels = max_channels
		self.out_channels = out_channels

		# Encoder
		self.enc_blocks = nn.ModuleList()
		channels: List[int] = []
		in_c = in_channels + 2
		for level in range(num_levels):
			out_c = min(base_channels * (2 ** level), max_channels)
			self.enc_blocks.append(DoubleConv(in_c, out_c))
			channels.append(out_c)
			in_c = out_c

		self.pool = nn.MaxPool2d(2)

		# Decoder
		self.up_convs = nn.ModuleList()
		self.dec_blocks = nn.ModuleList()
		for level in reversed(range(num_levels - 1)):
			in_c = channels[level + 1]
			out_c = channels[level]
			self.up_convs.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
			self.dec_blocks.append(DoubleConv(out_c * 2, out_c))

		self.out_head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		n, _c, h, w = x.shape
		device = x.device
		dtype = x.dtype
		yy = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(n, 1, h, w)
		xx = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(n, 1, h, w)
		x = torch.cat([x, xx, yy], dim=1)

		enc_feats: List[torch.Tensor] = []

		pool_count = 0
		for level, enc in enumerate(self.enc_blocks):
			x = enc(x)
			enc_feats.append(x)

			if level == self.num_levels - 1:
				break

			if x.size(2) >= 2 and x.size(3) >= 2:
				x = self.pool(x)
				pool_count += 1
			else:
				break

		effective_levels = len(enc_feats)
		if effective_levels < 1:
			raise RuntimeError("UNet encoder produced no feature maps")

		num_ups_to_use = max(0, effective_levels - 1)
		start_idx = (self.num_levels - 1) - num_ups_to_use

		for idx in range(num_ups_to_use):
			up = self.up_convs[start_idx + idx]
			dec = self.dec_blocks[start_idx + idx]

			x = up(x)

			skip_level = effective_levels - 2 - idx
			skip = enc_feats[skip_level]

			if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
				diff_h = skip.size(2) - x.size(2)
				diff_w = skip.size(3) - x.size(3)

				if diff_h > 0 or diff_w > 0:
					pad_top = diff_h // 2 if diff_h > 0 else 0
					pad_bottom = diff_h - pad_top if diff_h > 0 else 0
					pad_left = diff_w // 2 if diff_w > 0 else 0
					pad_right = diff_w - pad_left if diff_w > 0 else 0
					x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
				elif diff_h < 0 or diff_w < 0:
					crop_top = (-diff_h) // 2 if diff_h < 0 else 0
					crop_left = (-diff_w) // 2 if diff_w < 0 else 0
					x = x[
						:,
						:,
						crop_top : crop_top + skip.size(2),
						crop_left : crop_left + skip.size(3),
					]

			x = torch.cat([x, skip], dim=1)
			x = dec(x)

		feat = x
		out = self.out_head(feat)
		return out


def load_unet(
	device: Union[str, torch.device],
	weights: Optional[str] = None,
	in_channels: int = 1,
	out_channels: int = 3,
	base_channels: int = 32,
	num_levels: int = 6,
	max_channels: int = 1024,
) -> UNet:
	"""
	Construct a UNet and optionally load a checkpoint, filtering for matching keys.
	"""
	if isinstance(device, str):
		dev = torch.device(device)
	else:
		dev = device

	model = UNet(
		in_channels=in_channels,
		out_channels=out_channels,
		base_channels=base_channels,
		num_levels=num_levels,
		max_channels=max_channels,
	).to(dev)

	if weights is not None:
		ckpt = torch.load(weights, map_location=dev)
		if isinstance(ckpt, dict) and "state_dict" in ckpt:
			state_dict = ckpt["state_dict"]
		else:
			state_dict = ckpt

		model_state = model.state_dict()
		filtered = {}
		for k, v in state_dict.items():
			if k in model_state and model_state[k].shape == v.shape:
				filtered[k] = v
		model_state.update(filtered)
		missing_keys = [k for k in model_state.keys() if k not in filtered]
		unexpected_keys = [k for k in state_dict.keys() if k not in model_state]
		model.load_state_dict(model_state)
		if missing_keys:
			print(f"[load_unet] keeping randomly initialized params for missing keys: {sorted(missing_keys)}")
		if unexpected_keys:
			print(f"[load_unet] ignored unexpected checkpoint keys: {sorted(unexpected_keys)}")

	return model