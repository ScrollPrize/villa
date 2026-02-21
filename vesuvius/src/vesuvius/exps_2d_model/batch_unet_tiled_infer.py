from pathlib import Path
import argparse
from typing import Optional, List

import tifffile
import numpy as np
import torch

from common import load_unet, unet_infer_tiled


def _list_tiff_files(in_dir: Path) -> List[Path]:
	files = sorted(p for p in in_dir.glob("*.tif") if p.is_file())
	if not files:
		raise ValueError(f"No .tif files found in {in_dir}")
	return files


def _ensure_out_dir(out_dir: Path) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)


def _num_layers(path: Path) -> int:
	with tifffile.TiffFile(str(path)) as tif:
		series = tif.series[0]
		shape = series.shape
		if len(shape) == 2:
			return 1
		if len(shape) == 3:
			return int(shape[0])
		raise ValueError(f"Unsupported TIFF shape {shape} for {path}")


def _load_tiff_layer_uint8_norm(path: Path, layer: Optional[int], device: torch.device) -> torch.Tensor:
	"""
	Load a single TIFF layer as (1,1,H,W) float32 in [0,1], mirroring training preprocessing.

	- If stack is 2D, ignore `layer` and load that image.
	- If stack is 3D, use the given `layer` index (default 0 if None).
	- If dtype is uint16: downscale to uint8 via division by 257.
	"""
	with tifffile.TiffFile(str(path)) as tif:
		series = tif.series[0]
		shape = series.shape
		if len(shape) == 2:
			img = series.asarray()
		elif len(shape) == 3:
			idx = 0 if layer is None else int(layer)
			img = series.asarray(key=idx)
		else:
			raise ValueError(f"Unsupported TIFF shape {shape} for {path}")

	if img.dtype == np.uint16:
		img = (img // 257).astype(np.uint8)

	img_t = torch.from_numpy(img.astype("float32"))
	max_val = float(img_t.max())
	if max_val > 0.0:
		img_t = img_t / max_val
	img_t = img_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
	return img_t.to(device)


def batch_unet_tiled_infer(
	in_dir: Path,
	out_dir: Path,
	unet_checkpoint: str,
	device: Optional[str] = None,
	img_step: int = 1,
	layer_step: int = 1,
	tile_size: int = 512,
	overlap: int = 128,
) -> None:
	"""
	Run UNet inference with overlapping 2D tiles on all TIFF stacks/layers in a directory.
 
	For each input TIFF:
	- Iterate layers with given `layer_step`.
	- Run tiled UNet inference on the chosen layer.
	- Save four float32 TIFFs with suffixes:
	  - _cos.tif   : channel 0 (cosine branch)
	  - _mag.tif   : channel 1 (gradient magnitude branch)
	  - _dir.tif   : channel 2 (direction branch, 0.5 + 0.5*cos(2*theta))
	  - _dir2.tif  : channel 3 (direction branch, 0.5 + 0.5*cos(2*theta + pi/4))
	"""
	if img_step <= 0 or layer_step <= 0:
		raise ValueError("img_step and layer_step must be >= 1")

	in_dir = in_dir.resolve()
	out_dir = out_dir.resolve()
	_ensure_out_dir(out_dir)

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	torch_device = torch.device(device)

	model = load_unet(
		device=torch_device,
		weights=unet_checkpoint,
		in_channels=1,
		out_channels=4,
		base_channels=32,
		num_levels=6,
		max_channels=1024,
	)
	model.eval()

	files = _list_tiff_files(in_dir)

	for file_idx, tif_path in enumerate(files):
		if file_idx % img_step != 0:
			continue

		num_layers = _num_layers(tif_path)
		stem = tif_path.stem

		for layer_idx in range(0, num_layers, layer_step):
			prefix = out_dir / f"{stem}_layer{layer_idx:04d}"
			print(
				f"[{file_idx+1}/{len(files)}] {tif_path.name} "
				f"layer {layer_idx} -> {prefix}"
			)

			raw = _load_tiff_layer_uint8_norm(
				tif_path,
				layer=layer_idx,
				device=torch_device,
			)

			with torch.no_grad():
				pred = unet_infer_tiled(
					model,
					raw,
					tile_size=tile_size,
					overlap=overlap,
				)  # (1,C,H,W)
	
			pred_np = pred[0].detach().cpu().numpy().astype("float32")  # (C,H,W)
	
			cos_np = pred_np[0]
			mag_np = pred_np[1] if pred_np.shape[0] > 1 else None
			dir_np = pred_np[2] if pred_np.shape[0] > 2 else None
			dir2_np = pred_np[3] if pred_np.shape[0] > 3 else None
	
			tifffile.imwrite(f"{prefix}_cos.tif", cos_np, compression="lzw")
			if mag_np is not None:
				tifffile.imwrite(f"{prefix}_mag.tif", mag_np, compression="lzw")
			if dir_np is not None:
				tifffile.imwrite(f"{prefix}_dir.tif", dir_np, compression="lzw")
			if dir2_np is not None:
				tifffile.imwrite(f"{prefix}_dir2.tif", dir2_np, compression="lzw")


def main() -> None:
	parser = argparse.ArgumentParser(
		"Batch tiled UNet inference with overlapping 2D tiles blended linearly."
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
		help="Output directory for per-layer UNet predictions.",
	)
	parser.add_argument(
		"--unet-checkpoint",
		type=str,
		required=True,
		help="Path to UNet checkpoint (.pt) to use for inference.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help='Device string for inference, e.g. "cuda" or "cpu". Defaults to CUDA if available.',
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
		"--tile-size",
		type=int,
		default=512,
		help="Square tile size for tiled UNet inference (pixels).",
	)
	parser.add_argument(
		"--overlap",
		type=int,
		default=128,
		help="Overlap between neighboring tiles in pixels (per axis).",
	)

	args = parser.parse_args()

	in_dir = Path(args.in_dir)
	out_dir = Path(args.out_dir)

	batch_unet_tiled_infer(
		in_dir=in_dir,
		out_dir=out_dir,
		unet_checkpoint=args.unet_checkpoint,
		device=args.device,
		img_step=args.img_step,
		layer_step=args.layer_step,
		tile_size=args.tile_size,
		overlap=args.overlap,
	)


if __name__ == "__main__":
	main()