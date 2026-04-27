from pathlib import Path
import argparse
from typing import Optional

import tifffile
import numpy as np
import torch
import cv2

from common import load_unet, unet_infer_tiled


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


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Convert a float32 image array to uint8 [0,255] for fast JPG visualization.
    """
    if arr.size == 0:
        return np.zeros_like(arr, dtype="uint8")

    if np.issubdtype(arr.dtype, np.floating):
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax > vmin:
            norm = (arr - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(arr, dtype="float32")
        return (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")

    # Fallback: treat as already scaled numeric data.
    arr_f = arr.astype("float32")
    vmin = float(arr_f.min())
    vmax = float(arr_f.max())
    if vmax > vmin:
        norm = (arr_f - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(arr_f, dtype="float32")
    return (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")


def _load_omezarr_z_uint8_norm(
    *,
    path: str,
    z: int | None,
    crop: tuple[int, int, int, int] | None,
    device: torch.device,
) -> torch.Tensor:
    """Load one z-slice from an OME-Zarr array as (1,1,H,W) float32 in [0,1].

    Requirements:
    - `path` must point directly to a zarr array (not a group).
    - array must have exactly 3 dims: (Z,Y,X).
    """
    try:
        import zarr  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("OME-Zarr loading requires the 'zarr' package") from e

    a = zarr.open(str(path), mode="r")
    if not hasattr(a, "shape"):
        raise ValueError(f"OME-Zarr path must point to an array, got non-array: {path}")
    sh = tuple(int(v) for v in a.shape)
    if len(sh) != 3:
        raise ValueError(f"OME-Zarr array must have shape (Z,Y,X), got {sh} for {path}")

    zz = 0 if z is None else int(z)
    if zz < 0 or zz >= sh[0]:
        raise ValueError(f"z out of range: z={zz} shape={sh} for {path}")

    x0 = 0
    y0 = 0
    x1 = sh[2]
    y1 = sh[1]
    if crop is not None:
        x, y, w, h = (int(v) for v in crop)
        x0 = max(0, min(x, sh[2]))
        y0 = max(0, min(y, sh[1]))
        x1 = max(x0, min(x + w, sh[2]))
        y1 = max(y0, min(y + h, sh[1]))

    img = np.asarray(a[zz, y0:y1, x0:x1])
    # Mirror `_load_tiff_layer_uint8_norm` semantics: uint16 -> uint8 via //257,
    # then divide by the per-slice max (not the dtype max).
    if img.dtype == np.uint16:
        img = (img // 257).astype(np.uint8)
    img_t = torch.from_numpy(img.astype("float32"))
    max_val = float(img_t.max())
    if max_val > 0.0:
        img_t = img_t / max_val
    img_t = img_t.unsqueeze(0).unsqueeze(0)
    return img_t.to(device)


def tiled_unet_infer(
    image_path: str,
    out_dir: str,
    unet_checkpoint: str,
    device: Optional[str] = None,
    layer: Optional[int] = None,
    tile_size: int = 512,
    overlap: int = 128,
    border: int = 0,
) -> None:
    """
    Run UNet inference with overlapping 2D tiles on a single TIFF image/stack.
 
    border:
        Number of pixels at each tile border that are fully discarded (blend
        weight 0) before linear ramping begins. Useful to drop unreliable
        boundary predictions from each tile.
 
    Saves four float32 TIFFs (LZW) in `out_dir`:
      - <stem>[_layerXXXX]_cos.tif   : channel 0 (cosine branch)
      - ..._mag.tif                 : channel 1 (gradient magnitude branch)
      - ..._dir0.tif                : channel 2 (direction branch, 0.5 + 0.5*cos(2*theta))
      - ..._dir1.tif                : channel 3 (direction branch, 0.5 + 0.5*cos(2*theta + pi/4))
 
    Additionally saves 8-bit JPGs for fast visualization with the same base names
    (where applicable).
    """
    img_path = Path(image_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    raw = _load_tiff_layer_uint8_norm(
        img_path,
        layer=layer,
        device=torch_device,
    )

    model = load_unet(
        device=torch_device,
        weights=unet_checkpoint,
        strict=True,
        in_channels=1,
        out_channels=4,
        base_channels=32,
        num_levels=6,
        max_channels=1024,
    )
    model.eval()

    with torch.no_grad():
        pred = unet_infer_tiled(
            model,
            raw,
            tile_size=tile_size,
            overlap=overlap,
            border=border,
        )  # (1,3,H,W)

    pred_np = pred[0].detach().cpu().numpy().astype("float32")  # (C,H,W)

    if layer is not None:
        base = f"{img_path.stem}_layer{int(layer):04d}"
    else:
        base = img_path.stem
    prefix = out_dir_path / base

    cos_np = pred_np[0]
    mag_np = pred_np[1] if pred_np.shape[0] > 1 else None
    dir0_np = pred_np[2] if pred_np.shape[0] > 2 else None
    dir1_np = pred_np[3] if pred_np.shape[0] > 3 else None

    # Float32 TIFFs.
    tifffile.imwrite(f"{prefix}_cos.tif", cos_np, compression="lzw")
    if mag_np is not None:
        tifffile.imwrite(f"{prefix}_mag.tif", mag_np, compression="lzw")
    if dir0_np is not None:
        tifffile.imwrite(f"{prefix}_dir0.tif", dir0_np, compression="lzw")
    if dir1_np is not None:
        tifffile.imwrite(f"{prefix}_dir1.tif", dir1_np, compression="lzw")

    # 8-bit JPGs for fast visualization.
    cos_u8 = _to_uint8(cos_np)
    cv2.imwrite(str(prefix) + "_cos.jpg", cos_u8)
    if mag_np is not None:
        mag_u8 = _to_uint8(mag_np)
        cv2.imwrite(str(prefix) + "_mag.jpg", mag_u8)
    if dir0_np is not None:
        dir0_u8 = _to_uint8(dir0_np)
        cv2.imwrite(str(prefix) + "_dir0.jpg", dir0_u8)
    if dir1_np is not None:
        dir1_u8 = _to_uint8(dir1_np)
        cv2.imwrite(str(prefix) + "_dir1.jpg", dir1_u8)


def main() -> None:
    parser = argparse.ArgumentParser(
        "Tiled UNet inference with overlapping 2D tiles blended linearly."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input TIFF image / stack.",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=str,
        required=True,
        help="Path to UNet checkpoint (.pt) to use for inference.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for TIFF/JPG predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string for inference, e.g. "cuda" or "cpu". Defaults to CUDA if available.',
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index of the input TIFF stack (ignored for 2D TIFFs).",
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
    parser.add_argument(
        "--border",
        type=int,
        default=0,
        help="Border pixels per tile to discard completely (blend weight 0).",
    )

    args = parser.parse_args()

    tiled_unet_infer(
        image_path=args.image,
        out_dir=args.out_dir,
        unet_checkpoint=args.unet_checkpoint,
        device=args.device,
        layer=args.layer,
        tile_size=args.tile_size,
        overlap=args.overlap,
        border=args.border,
    )


if __name__ == "__main__":
    main()
