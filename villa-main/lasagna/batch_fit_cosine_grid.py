from pathlib import Path
import argparse
from typing import Optional, List

import tifffile

from fit_cosine_grid import fit_cosine_grid


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


def batch_fit_cosine(
	in_dir: Path,
	out_dir: Path,
	img_step: int = 1,
	layer_step: int = 1,
	steps: int = 1000,
	steps_stage1: int = 500,
	lr: float = 1e-2,
	grid_step: int = 4,
	output_scale: int = 4,
	cosine_periods: float = 32.0,
	sample_scale: float = 1.0,
	samples_per_period: float = 1.0,
	dense_samples_per_period: float = 8.0,
	img_downscale_factor: float = 2.0,
	unet_checkpoint: Optional[str] = None,
	device: Optional[str] = None,
	compile_model: bool = False,
) -> None:
	if img_step <= 0 or layer_step <= 0:
		raise ValueError("img_step and layer_step must be >= 1")

	in_dir = in_dir.resolve()
	out_dir = out_dir.resolve()
	_ensure_out_dir(out_dir)

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

			fit_cosine_grid(
				image_path=str(tif_path),
				steps=steps,
				steps_stage1=steps_stage1,
				lr=lr,
				grid_step=grid_step,
				device=device,
				output_prefix=str(prefix),
				snapshot=None,
				output_scale=output_scale,
				dbg=False,
				mask_cx=None,
				mask_cy=None,
				cosine_periods=cosine_periods,
				sample_scale=sample_scale,
				samples_per_period=samples_per_period,
				dense_samples_per_period=dense_samples_per_period,
				img_downscale_factor=img_downscale_factor,
				for_video=False,
				unet_checkpoint=unet_checkpoint,
				unet_layer=layer_idx,
				unet_crop=8,
				compile_model=compile_model,
				final_float=True,
			)


def main() -> None:
	parser = argparse.ArgumentParser(
		"Batch fit 2D cosine grid to all TIFF stacks and layers in a directory."
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
		help="Output directory for per-layer results.",
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
		"--steps",
		type=int,
		default=1000,
		help="Number of optimization steps for stage 2 (joint).",
	)
	parser.add_argument(
		"--steps-stage1",
		type=int,
		default=500,
		help="Number of optimization steps for stage 1 (global rotation + scale).",
	)
	parser.add_argument("--lr", type=float, default=1e-2)
	parser.add_argument(
		"--grid-step",
		type=int,
		default=4,
		help="Vertical coarse grid step in sample-space pixels.",
	)
	parser.add_argument(
		"--output-scale",
		type=int,
		default=4,
		help="Integer scale factor for saving reconstructions.",
	)
	parser.add_argument(
		"--cosine-periods",
		type=float,
		default=32.0,
		help="Number of cosine periods across the sample-space width.",
	)
	parser.add_argument(
		"--sample-scale",
		type=float,
		default=1.0,
		help="Global multiplier for internal sample-space resolution.",
	)
	parser.add_argument(
		"--samples-per-period",
		type=float,
		default=1.0,
		help="Number of coarse grid steps per cosine period horizontally.",
	)
	parser.add_argument(
		"--dense-samples-per-period",
		type=float,
		default=8.0,
		help="Dense samples per cosine period for the internal x resolution.",
	)
	parser.add_argument(
		"--img-downscale-factor",
		type=float,
		default=2.0,
		help="Downscale factor for internal resolution relative to avg image size.",
	)
	parser.add_argument(
		"--unet-checkpoint",
		type=str,
		default=None,
		help=(
			"Optional UNet checkpoint (.pt); if set, run UNet and fit cosine "
			"grid to its channel-0 output instead of the raw image."
		),
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help='Device string for fitting, e.g. "cuda" or "cpu". Defaults to CUDA if available.',
	)
	parser.add_argument(
		"--compile-model",
		action="store_true",
		help="Compile CosineGridModel with torch.compile (PyTorch 2.x) for faster training.",
	)

	args = parser.parse_args()

	in_dir = Path(args.in_dir)
	out_dir = Path(args.out_dir)
	unet_checkpoint = args.unet_checkpoint if args.unet_checkpoint is not None else None

	batch_fit_cosine(
		in_dir=in_dir,
		out_dir=out_dir,
		img_step=args.img_step,
		layer_step=args.layer_step,
		steps=args.steps,
		steps_stage1=args.steps_stage1,
		lr=args.lr,
		grid_step=args.grid_step,
		output_scale=args.output_scale,
		cosine_periods=args.cosine_periods,
		sample_scale=args.sample_scale,
		samples_per_period=args.samples_per_period,
		dense_samples_per_period=args.dense_samples_per_period,
		img_downscale_factor=args.img_downscale_factor,
		unet_checkpoint=unet_checkpoint,
		device=args.device,
		compile_model=args.compile_model,
	)


if __name__ == "__main__":
	main()