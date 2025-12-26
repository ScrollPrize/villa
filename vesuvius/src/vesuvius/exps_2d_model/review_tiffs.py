from pathlib import Path
import argparse
from typing import Optional, List

import numpy as np
import tifffile
import torch
from PIL import Image

from common import load_unet


def _list_tiff_files(in_dir: Path) -> List[Path]:
	files = sorted(p for p in in_dir.glob("*.tif") if p.is_file())
	if not files:
		raise ValueError(f"No .tif files found in {in_dir}")
	return files


def _ensure_out_dir(out_dir: Path) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)


def _load_stack(path: Path) -> np.ndarray:
	"""Load a TIFF stack as (L,H,W) or (1,H,W)."""
	arr = tifffile.imread(str(path))
	if arr.ndim == 2:
		arr = arr[None, ...]
	elif arr.ndim != 3:
		raise ValueError(f"Unsupported TIFF ndim={arr.ndim} for {path}")
	return arr


def _to_uint8_image(layer: np.ndarray) -> np.ndarray:
	"""Convert a single 2D layer to uint8 [0,255] for JPEG saving."""
	if layer.dtype == np.uint8:
		return layer
	if layer.dtype == np.uint16:
		return (layer // 257).astype(np.uint8)
	layer_f = layer.astype(np.float32)
	# Try to infer scale; fallback to min-max.
	vmin = float(layer_f.min())
	vmax = float(layer_f.max())
	if vmax <= vmin:
		return np.zeros_like(layer_f, dtype=np.uint8)
	layer_norm = (layer_f - vmin) / (vmax - vmin)
	layer_norm = np.clip(layer_norm, 0.0, 1.0)
	return (layer_norm * 255.0 + 0.5).astype(np.uint8)


def _save_jpeg_gray(img: np.ndarray, path: Path) -> None:
	img_u8 = _to_uint8_image(img)
	im = Image.fromarray(img_u8, mode="L")
	im.save(str(path), format="JPEG")


def _prepare_input_tensor(layer: np.ndarray, device: torch.device) -> torch.Tensor:
	"""Convert a 2D numpy layer to model input tensor (1,1,H,W) in [0,1]."""
	if layer.dtype == np.uint16:
		layer_u8 = (layer // 257).astype(np.uint8)
		x = torch.from_numpy(layer_u8.astype("float32")) / 255.0
	elif layer.dtype == np.uint8:
		x = torch.from_numpy(layer.astype("float32")) / 255.0
	else:
		layer_f = layer.astype(np.float32)
		# Assume already roughly in [0,1] but clip for safety.
		layer_f = np.clip(layer_f, 0.0, 1.0)
		x = torch.from_numpy(layer_f)
	x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
	return x.to(device)


def _tensor_to_display_uint8(t: torch.Tensor) -> np.ndarray:
	"""Convert (1,H,W) tensor to uint8 image via per-image min-max."""
	if t.ndim != 3 or t.size(0) != 1:
		raise ValueError(f"Expected tensor shape (1,H,W), got {tuple(t.shape)}")
	arr = t.squeeze(0).cpu().numpy().astype(np.float32)
	vmin = float(arr.min())
	vmax = float(arr.max())
	if vmax <= vmin:
		return np.zeros_like(arr, dtype=np.uint8)
	arr_norm = (arr - vmin) / (vmax - vmin)
	arr_norm = np.clip(arr_norm, 0.0, 1.0)
	return (arr_norm * 255.0 + 0.5).astype(np.uint8)


def _run_unet_on_layer(
	model: torch.nn.Module,
	device: torch.device,
	layer: np.ndarray,
) -> List[np.ndarray]:
	"""
	Run UNet on a single 2D layer and return three uint8 prediction images
	(cos, mag, dir) as HxW arrays.
	"""
	model.eval()
	with torch.no_grad():
		x = _prepare_input_tensor(layer, device)
		pred = model(x)  # (1,3,H,W)
		if pred.ndim != 4 or pred.size(1) < 3:
			raise ValueError(f"Expected model output (1,3,H,W), got {tuple(pred.shape)}")
		chans: List[np.ndarray] = []
		for c in range(3):
			ch = pred[:, c : c + 1, :, :]  # (1,1,H,W)
			ch_u8 = _tensor_to_display_uint8(ch.squeeze(1))
			chans.append(ch_u8)
	return chans


def review_tiffs(
	in_dir: Path,
	out_dir: Path,
	img_step: int = 1,
	layer_step: int = 1,
	checkpoint: Optional[Path] = None,
	device_str: Optional[str] = None,
) -> None:
	if img_step <= 0 or layer_step <= 0:
		raise ValueError("img_step and layer_step must be >= 1")

	in_dir = in_dir.resolve()
	out_dir = out_dir.resolve()
	_ensure_out_dir(out_dir)

	files = _list_tiff_files(in_dir)

	if device_str is None:
		device_str = "cuda" if torch.cuda.is_available() else "cpu"
	device = torch.device(device_str)

	model: Optional[torch.nn.Module] = None
	if checkpoint is not None:
		ckpt_path = checkpoint.resolve()
		if not ckpt_path.is_file():
			raise ValueError(f"Checkpoint not found: {ckpt_path}")
		model = load_unet(
			device=device_str,
			weights=str(ckpt_path),
			in_channels=1,
			out_channels=3,
			base_channels=32,
			num_levels=6,
			max_channels=1024,
		)

	for file_idx, tif_path in enumerate(files):
		if (file_idx % img_step) != 0:
			continue

		stack = _load_stack(tif_path)  # (L,H,W)
		num_layers = stack.shape[0]
		stem = tif_path.stem

		for layer_idx in range(0, num_layers, layer_step):
			layer = stack[layer_idx]

			base_name = f"{stem}_layer{layer_idx:04d}"
			jpg_path = out_dir / f"{base_name}.jpg"
			_save_jpeg_gray(layer, jpg_path)

			if model is not None:
				cos_img, mag_img, dir_img = _run_unet_on_layer(model, device, layer)
				_save_jpeg_gray(cos_img, out_dir / f"{base_name}_cos.jpg")
				_save_jpeg_gray(mag_img, out_dir / f"{base_name}_mag.jpg")
				_save_jpeg_gray(dir_img, out_dir / f"{base_name}_dir.jpg")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Review multi-layer TIFF stacks by exporting layers to JPEG and optionally running UNet inference.",
	)
	parser.add_argument(
		"--in-dir",
		type=str,
		required=True,
		help="Input directory containing .tif stacks.",
	)
	parser.add_argument(
		"--out-dir",
		type=str,
		required=True,
		help="Output directory for JPEGs.",
	)
	parser.add_argument(
		"--img-step",
		type=int,
		default=1,
		help="Process every N-th TIFF file (default: 1).",
	)
	parser.add_argument(
		"--layer-step",
		type=int,
		default=1,
		help="Process every N-th layer within each TIFF (default: 1).",
	)
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=None,
		help="Optional UNet checkpoint (.pt); if set, run inference and save cos/mag/dir JPEGs.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help='Device string for inference, e.g. "cuda" or "cpu". Defaults to CUDA if available.',
	)

	args = parser.parse_args()

	in_dir = Path(args.in_dir)
	out_dir = Path(args.out_dir)
	checkpoint = Path(args.checkpoint) if args.checkpoint is not None else None

	review_tiffs(
		in_dir=in_dir,
		out_dir=out_dir,
		img_step=args.img_step,
		layer_step=args.layer_step,
		checkpoint=checkpoint,
		device_str=args.device,
	)


if __name__ == "__main__":
	main()