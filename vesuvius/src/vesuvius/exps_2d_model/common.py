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

def unet_infer_tiled(
	model: UNet,
	image: torch.Tensor,
	tile_size: int = 512,
	overlap: int = 128,
) -> torch.Tensor:
	"""
	Run 2D UNet inference on an image using overlapping tiles with linear blending.

	Args:
		model:
			UNet model. Assumed to be in eval() mode & run under no_grad() by caller.
		image:
			Input tensor of shape (N,C,H,W). Typically N=1, C=1 as used in this repo.
		tile_size:
			Spatial size of square tiles (tile_size x tile_size).
		overlap:
			Number of pixels of linear-overlap between neighboring tiles along each
			axis. Effective stride is (tile_size - overlap). Must be < tile_size.

	Returns:
		Tensor of shape (N, model.out_channels, H, W) with stitched prediction.
	"""
	if image.ndim != 4:
		raise ValueError(f"expected image of shape (N,C,H,W), got {tuple(image.shape)}")

	n, c, h, w = image.shape
	if n > 1:
		# Process batch element-wise to keep tiling/simple.
		outs = []
		for i in range(n):
			outs.append(unet_infer_tiled(model, image[i : i + 1], tile_size=tile_size, overlap=overlap))
		return torch.cat(outs, dim=0)

	if tile_size <= 0:
		raise ValueError(f"tile_size must be > 0, got {tile_size}")
	if overlap < 0 or overlap >= tile_size:
		raise ValueError(f"overlap must satisfy 0 <= overlap < tile_size (got {overlap}, tile_size={tile_size})")

	# Small images: fall back to single forward pass.
	if h <= tile_size and w <= tile_size and overlap == 0:
		return model(image)

	device = image.device
	dtype = image.dtype

	def _build_positions(size: int, tile: int, stride: int) -> list[int]:
		if size <= tile:
			return [0]
		positions = list(range(0, size - tile + 1, stride))
		last = size - tile
		if positions[-1] != last:
			positions.append(last)
		return positions

	stride = max(1, tile_size - overlap)
	y_positions = _build_positions(h, tile_size, stride)
	x_positions = _build_positions(w, tile_size, stride)

	# Precompute separable 2D blending mask for a tile.
	def _blend_ramp(length: int, ov: int) -> torch.Tensor:
		if ov <= 0:
			return torch.ones(length, device=device, dtype=dtype)
		ov = min(ov, length // 2)
		ramp = torch.ones(length, device=device, dtype=dtype)
		if ov > 0:
			# Linear ramps towards the borders; interior stays at 1.
			# Values at the outermost ov pixels go from ~0 -> 1 and 1 -> ~0,
			# but since both numerator & denominator are weighted the same,
			# absolute scale cancels after division.
			edges = torch.linspace(0.0, 1.0, steps=ov + 1, device=device, dtype=dtype)[1:]
			ramp[:ov] = edges
			ramp[-ov:] = edges.flip(0)
		return ramp

	ramp_y = _blend_ramp(tile_size, overlap)
	ramp_x = _blend_ramp(tile_size, overlap)
	weight_tile = (ramp_y.view(-1, 1) * ramp_x.view(1, -1)).unsqueeze(0).unsqueeze(0)  # (1,1,Th,Tw)

	# Accumulators for blended output & weights.
	out_channels = getattr(model, "out_channels", None)
	if out_channels is None:
		# Fallback: run a tiny dummy forward to infer channel count.
		with torch.no_grad():
			dummy = torch.zeros(1, c, min(tile_size, h), min(tile_size, w), device=device, dtype=dtype)
			out_channels = model(dummy).shape[1]

	acc = torch.zeros(1, out_channels, h, w, device=device, dtype=dtype)
	wsum = torch.zeros(1, 1, h, w, device=device, dtype=dtype)

	for y0 in y_positions:
		for x0 in x_positions:
			y1 = min(y0 + tile_size, h)
			x1 = min(x0 + tile_size, w)
			patch = image[:, :, y0:y1, x0:x1]
			ph = y1 - y0
			pw = x1 - x0

			# Crop weight mask if we are at the image boundary.
			w_patch = weight_tile[:, :, :ph, :pw]

			with torch.no_grad():
				pred = model(patch)  # (1,out_channels,ph,pw)

			acc[:, :, y0:y1, x0:x1] += pred * w_patch
			wsum[:, :, y0:y1, x0:x1] += w_patch

	# Avoid division by zero; outside any tile wsum is 0, but that should not occur.
	eps = torch.finfo(dtype).eps if torch.is_floating_point(wsum) else 1e-6
	wsafe = torch.clamp(wsum, min=eps)
	out = acc / wsafe
	return out