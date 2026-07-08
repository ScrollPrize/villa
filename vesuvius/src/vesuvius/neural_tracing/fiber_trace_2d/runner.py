from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.fiber_trace_2d.loader import FiberStrip2DLoader, load_config


def _to_u8_image(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if not bool(valid.any()):
        return out
    values = arr[valid]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        out[valid] = 127
        return out
    scaled = (arr - lo) * (255.0 / (hi - lo))
    out[valid] = np.clip(scaled[valid], 0.0, 255.0).astype(np.uint8)
    return out


def _write_jpg(path: Path, image_u8: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(image_u8, dtype=np.uint8), mode="L").save(path, quality=95)


def _write_contact_sheet(path: Path, images: list[np.ndarray], *, columns: int = 8) -> None:
    if not images:
        return
    h, w = images[0].shape
    cols = max(1, min(columns, len(images)))
    rows = (len(images) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        sheet[row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    _write_jpg(path, sheet)


def _offset_label(offset: float) -> str:
    value = float(offset)
    if value.is_integer():
        return f"{int(value):+04d}"
    return f"{value:+07.3f}".replace(".", "p")


def _batch_control_points(batch) -> list[tuple[int, np.ndarray]]:
    points: list[tuple[int, np.ndarray]] = []
    offsets_per_sample = int(len(batch.strip_z_offsets))
    for batch_index in range(batch.images.shape[0]):
        sample = batch.samples[batch_index * offsets_per_sample]
        points.append((int(sample.control_point_index), np.asarray(sample.control_point_xyz, dtype=np.float64)))
    return points


def _export_batch(batch, output_dir: str | Path) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    image_files: list[str] = []
    mask_files: list[str] = []
    contact_images: list[np.ndarray] = []
    planar_image_files: list[str] = []
    planar_mask_files: list[str] = []
    planar_contact_images: list[np.ndarray] = []
    for batch_index in range(batch.images.shape[0]):
        for offset_index, offset in enumerate(batch.strip_z_offsets.tolist()):
            stem = f"sample_{batch_index:03d}_offset_{_offset_label(offset)}"
            image_name = f"{stem}.jpg"
            mask_name = f"{stem}_valid.jpg"
            planar_image_name = f"{stem}_planar.jpg"
            planar_mask_name = f"{stem}_planar_valid.jpg"
            image = batch.images[batch_index, offset_index, 0]
            valid_mask = batch.valid_mask[batch_index, offset_index].astype(bool)
            image_u8 = _to_u8_image(image, valid_mask)
            mask_u8 = (valid_mask.astype(np.uint8) * 255)
            _write_jpg(out / image_name, image_u8)
            _write_jpg(out / mask_name, mask_u8)

            planar_image = batch.planar_images[batch_index, offset_index, 0]
            planar_valid_mask = batch.planar_valid_mask[batch_index, offset_index].astype(bool)
            planar_image_u8 = _to_u8_image(planar_image, planar_valid_mask)
            planar_mask_u8 = (planar_valid_mask.astype(np.uint8) * 255)
            _write_jpg(out / planar_image_name, planar_image_u8)
            _write_jpg(out / planar_mask_name, planar_mask_u8)

            contact_images.append(image_u8)
            planar_contact_images.append(planar_image_u8)
            image_files.append(image_name)
            mask_files.append(mask_name)
            planar_image_files.append(planar_image_name)
            planar_mask_files.append(planar_mask_name)

    _write_contact_sheet(out / "contact_sheet.jpg", contact_images)
    _write_contact_sheet(out / "contact_sheet_planar.jpg", planar_contact_images)
    summary_lines = [
        f"images_shape={list(batch.images.shape)}",
        f"coords_zyx_shape={list(batch.coords_zyx.shape)}",
        f"valid_mask_shape={list(batch.valid_mask.shape)}",
        f"planar_images_shape={list(batch.planar_images.shape)}",
        f"planar_coords_zyx_shape={list(batch.planar_coords_zyx.shape)}",
        f"planar_valid_mask_shape={list(batch.planar_valid_mask.shape)}",
        f"strip_z_offsets={batch.strip_z_offsets.tolist()}",
        f"record_indices={batch.record_indices.tolist()}",
        f"control_point_indices={batch.control_point_indices.tolist()}",
    ]
    for i, path in enumerate(batch.fiber_paths):
        summary_lines.append(f"sample_{i}_fiber_path={path}")
    for i, (cp_index, cp_xyz) in enumerate(_batch_control_points(batch)):
        cp_zyx = cp_xyz[[2, 1, 0]]
        summary_lines.append(
            f"sample_{i}_cp_index={cp_index} "
            f"cp_xyz=({cp_xyz[0]:.3f}, {cp_xyz[1]:.3f}, {cp_xyz[2]:.3f}) "
            f"cp_zyx=({cp_zyx[0]:.3f}, {cp_zyx[1]:.3f}, {cp_zyx[2]:.3f})"
        )
    (out / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(
        f"exported {len(image_files)} strip jpgs, {len(mask_files)} strip mask jpgs, "
        f"{len(planar_image_files)} planar jpgs, {len(planar_mask_files)} planar mask jpgs, "
        f"contact sheets, and summary.txt to {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load or prefetch 2D fiber-strip batches.")
    parser.add_argument("config", help="Path to Vesuvius-style JSON loader config")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--prefetch-samples", type=int, default=None)
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--export-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    loader = FiberStrip2DLoader(config)
    batch_size = config.batch_size if args.batch_size is None else args.batch_size

    if args.prefetch:
        sample_count = batch_size if args.prefetch_samples is None else args.prefetch_samples
        summary = loader.prefetch(args.sample_index, sample_count)
        print(
            "prefetch summary: "
            f"generated={summary['generated']} missing={summary['missing']} "
            f"downloaded={summary['downloaded']} bytes={summary['bytes']} "
            f"errors={summary['errors']} workers={summary['workers']}"
        )
        return

    if not args.skip_prefetch:
        loader.prefetch(args.sample_index, batch_size)

    batch = loader.load_batch(args.sample_index, batch_size=batch_size)
    print(f"images shape={batch.images.shape} dtype={batch.images.dtype}")
    print(f"coords_zyx shape={batch.coords_zyx.shape} dtype={batch.coords_zyx.dtype}")
    print(f"valid_mask shape={batch.valid_mask.shape} dtype={batch.valid_mask.dtype}")
    print(f"planar_images shape={batch.planar_images.shape} dtype={batch.planar_images.dtype}")
    print(f"planar_coords_zyx shape={batch.planar_coords_zyx.shape} dtype={batch.planar_coords_zyx.dtype}")
    print(f"planar_valid_mask shape={batch.planar_valid_mask.shape} dtype={batch.planar_valid_mask.dtype}")
    print(f"strip_z_offsets={batch.strip_z_offsets.tolist()}")
    print(f"record_indices={batch.record_indices.tolist()}")
    print(f"control_point_indices={batch.control_point_indices.tolist()}")
    for i, path in enumerate(batch.fiber_paths):
        print(f"sample {i}: fiber_path={path}")
    for i, (cp_index, cp_xyz) in enumerate(_batch_control_points(batch)):
        cp_zyx = cp_xyz[[2, 1, 0]]
        print(
            f"sample {i}: cp_index={cp_index} "
            f"cp_xyz=({cp_xyz[0]:.3f}, {cp_xyz[1]:.3f}, {cp_xyz[2]:.3f}) "
            f"cp_zyx=({cp_zyx[0]:.3f}, {cp_zyx[1]:.3f}, {cp_zyx[2]:.3f})"
        )
    stats = batch.cache_stats
    if stats is not None:
        print(
            "cache: "
            f"hits={getattr(stats, 'cache_hits', 0)} "
            f"downloads={getattr(stats, 'downloads', 0)} "
            f"missing={getattr(stats, 'missing', 0)}"
        )
    if args.export_dir:
        _export_batch(batch, args.export_dir)


if __name__ == "__main__":
    main()
