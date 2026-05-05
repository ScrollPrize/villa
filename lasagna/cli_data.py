from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

import fit_data


@dataclass(frozen=True)
class DataConfig:
	input: str
	device: str
	downscale: float
	crop: tuple[int, int, int, int, int, int] | None  # (x, y, z, w, h, d) fullres
	seed: tuple[int, int, int] | None                   # (cx, cy, cz) fullres seed point
	model_w: int | None                                  # model width in fullres voxels
	model_h: int | None                                  # model height in fullres voxels
	windings: int | None                                 # number of windings
	winding_volume: str | None                           # path to winding volume zarr
	cuda_gridsample: bool                                # use custom CUDA uint8 grid_sample kernel
	erode_valid_mask: int                                # erode grad_mag validity mask by N voxels
	sparse_prefetch_backend: str                         # "tensorstore_cpp" or "python" streaming prefetcher


def add_args(p: argparse.ArgumentParser) -> None:
	g = p.add_argument_group("data")
	g.add_argument("--input", default=None)
	g.add_argument("--device", default="cuda")
	g.add_argument("--downscale", type=float, default=4.0)
	g.add_argument("--crop", type=int, nargs=6, default=None,
		metavar=("X", "Y", "Z", "W", "H", "D"),
		help="3D volume crop in fullres voxels: x y z w h d")
	g.add_argument("--seed", type=int, nargs=3, default=None,
		metavar=("CX", "CY", "CZ"),
		help="Seed point in fullres voxels")
	g.add_argument("--model-w", type=int, default=None,
		help="Model width in fullres voxels")
	g.add_argument("--model-h", type=int, default=None,
		help="Model height in fullres voxels")
	g.add_argument("--windings", type=int, default=None,
		help="Number of windings")
	g.add_argument("--winding-volume", default=None,
		help="Path to winding volume zarr (float32, from labels_to_winding_volume.py)")
	g.add_argument("--cuda-gridsample", type=int, default=1,
		help="Use custom CUDA uint8 grid_sample kernel (1=yes, 0=fallback to PyTorch F.grid_sample)")
	g.add_argument("--erode-valid-mask", type=int, default=0,
		help="Erode grad_mag validity mask inward by N voxels (excludes noisy borders from all losses)")
	g.add_argument("--sparse-prefetch-backend", choices=("tensorstore_cpp", "python", "cuda", "tensorstore"), default="tensorstore_cpp",
		help="Sparse streaming prefetch backend: tensorstore_cpp uses TensorStore C++, python uses the old zarr path")


def from_args(args: argparse.Namespace) -> DataConfig:
	if args.input in (None, ""):
		raise ValueError("missing --input (can be provided via JSON config args)")
	crop = None
	if args.crop is not None:
		crop = tuple(int(v) for v in args.crop)
		if len(crop) != 6:
			raise ValueError("--crop requires exactly 6 values: x y z w h d")
	seed = None
	if getattr(args, "seed", None) is not None:
		seed = tuple(int(v) for v in args.seed)
		if len(seed) != 3:
			raise ValueError("--seed requires exactly 3 values: cx cy cz")
	model_w = None if getattr(args, "model_w", None) is None else int(args.model_w)
	model_h = None if getattr(args, "model_h", None) is None else int(args.model_h)
	windings = None if getattr(args, "windings", None) is None else int(args.windings)
	winding_volume = getattr(args, "winding_volume", None)
	if winding_volume is not None:
		winding_volume = str(winding_volume)
	return DataConfig(
		input=str(args.input),
		device=str(args.device),
		downscale=float(args.downscale),
		crop=crop,
		seed=seed,
		model_w=model_w,
		model_h=model_h,
		windings=windings,
		winding_volume=winding_volume,
		cuda_gridsample=bool(int(getattr(args, "cuda_gridsample", 1))),
		erode_valid_mask=int(getattr(args, "erode_valid_mask", 0)),
		sparse_prefetch_backend=str(getattr(args, "sparse_prefetch_backend", "tensorstore_cpp")),
	)


def load_fit_data(cfg: DataConfig) -> fit_data.FitData3D:
	return fit_data.load_3d_streaming(
		path=cfg.input,
		device=torch.device(cfg.device),
		sparse_prefetch_backend=cfg.sparse_prefetch_backend,
	)
