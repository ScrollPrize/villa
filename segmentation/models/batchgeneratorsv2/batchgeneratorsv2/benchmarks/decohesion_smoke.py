import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from batchgeneratorsv2.transforms.noise.extranoisetransforms import DecohesionTransform


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def _synthetic_volume(shape: tuple[int, int, int]) -> torch.Tensor:
    z, y, x = shape
    zz = torch.linspace(-1, 1, z)[:, None, None]
    yy = torch.linspace(-1, 1, y)[None, :, None]
    xx = torch.linspace(-1, 1, x)[None, None, :]
    fibers = torch.sin(42 * (xx + 0.4 * yy) + 9 * zz)
    sheet = torch.exp(-((yy + 0.25 * torch.sin(4 * zz)) ** 2) / 0.02)
    texture = 0.08 * torch.randn(shape)
    return (sheet * (0.7 + 0.3 * fibers) + texture).unsqueeze(0).float()


def _load_tiff_stack(layers_dir: Path, crop_size: int, max_slices: int) -> torch.Tensor:
    from tifffile import imread

    paths = sorted(p for p in layers_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"})
    if not paths:
        raise FileNotFoundError(f"No TIFF slices found in {layers_dir}")
    slices = []
    for path in paths[:max_slices]:
        arr = imread(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D TIFF slice, got shape {arr.shape} for {path}")
        y0 = max((arr.shape[0] - crop_size) // 2, 0)
        x0 = max((arr.shape[1] - crop_size) // 2, 0)
        slices.append(arr[y0:y0 + crop_size, x0:x0 + crop_size].astype(np.float32))
    stack = np.stack(slices, axis=0)
    stack -= float(stack.mean())
    stack /= float(stack.std() + 1e-6)
    return torch.from_numpy(stack).unsqueeze(0)


def _time_transform(transform: DecohesionTransform, image: torch.Tensor, repeats: int) -> float:
    if image.device.type == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(repeats):
        transform(image=image.clone())
    if image.device.type == "cuda":
        torch.cuda.synchronize()
    return (perf_counter() - start) / repeats


def _save_contact_sheet(before: torch.Tensor, after: torch.Tensor, path: Path) -> None:
    from PIL import Image, ImageDraw

    before_np = before.detach().cpu().numpy()[0]
    after_np = after.detach().cpu().numpy()[0]
    idxs = sorted(set([0, before_np.shape[0] // 2, before_np.shape[0] - 1]))
    tiles = []
    for label, stack in [("before", before_np), ("after", after_np)]:
        for idx in idxs:
            img = Image.fromarray(_normalize_to_uint8(stack[idx])).convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 92, 18), fill=(0, 0, 0))
            draw.text((4, 3), f"{label} z={idx}", fill=(255, 255, 255))
            tiles.append(img)
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (width * len(idxs), height * 2), (0, 0, 0))
    for n, tile in enumerate(tiles):
        row = 0 if n < len(idxs) else 1
        col = n % len(idxs)
        sheet.paste(tile, (col * width, row * height))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke benchmark for scroll decohesion augmentation.")
    parser.add_argument("--layers-dir", type=Path, help="Optional directory containing TIFF slices.")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-slices", type=int, default=9)
    parser.add_argument("--synthetic-shape", default="9,256,256")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("decohesion-smoke-out"))
    args = parser.parse_args()

    if args.layers_dir:
        image = _load_tiff_stack(args.layers_dir, args.crop_size, args.max_slices)
        source = str(args.layers_dir)
    else:
        image = _synthetic_volume(tuple(int(i) for i in args.synthetic_shape.split(",")))
        source = "synthetic"

    transform = DecohesionTransform(
        shift=(2, 0),
        alpha=0.65,
        num_prev_slices=2,
        smear_axis=1,
        p_per_channel=1.0,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cpu_time = _time_transform(transform, image.cpu(), args.repeats)
    result = transform(image=image.clone())["image"]
    _save_contact_sheet(image, result, args.out_dir / "decohesion_contact_sheet.png")

    report = {
        "source": source,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "cpu_seconds_per_run": cpu_time,
        "cuda_seconds_per_run": None,
        "contact_sheet": str(args.out_dir / "decohesion_contact_sheet.png"),
    }
    if torch.cuda.is_available():
        cuda_image = image.cuda()
        report["cuda_device"] = torch.cuda.get_device_name(0)
        report["cuda_seconds_per_run"] = _time_transform(transform, cuda_image, args.repeats)

    (args.out_dir / "decohesion_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
