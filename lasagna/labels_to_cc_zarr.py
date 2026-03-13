"""Read a multi-layer label TIFF (0=bg, 1=pred/fg, 2=ignore), run 3D
connected components on the foreground, store:
  - cc zarr: uint8 (0=bg, 1..N=components)
  - binary zarr: uint8 (0=bg+ignore, 1=fg)
and render a slice-video with random per-component colours.

Usage
-----
  python labels_to_cc_zarr.py \
      --input labels.tif \
      --output cc.zarr \
      --binary-zarr binary.zarr \
      --video cc_video.mp4 \
      --axis z \
      --connectivity 6
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import tifffile
import zarr


# ---------------------------------------------------------------------------
# Connected components (scipy preferred, fallback to cc3d if available)
# ---------------------------------------------------------------------------

def _connected_components(
    vol: np.ndarray,
    connectivity: int,
) -> tuple[np.ndarray, int]:
    """Label connected components in a binary 3D volume.

    Returns (labels, n_components).  *labels* dtype is int32/int64.
    *connectivity* is 6 (face) or 26 (full).
    """
    try:
        import cc3d  # faster, handles huge volumes well
        labels = cc3d.connected_components(vol, connectivity=connectivity)
        n = int(labels.max())
        return labels, n
    except ImportError:
        pass

    from scipy.ndimage import label as scipy_label

    if connectivity == 6:
        struct3d = np.zeros((3, 3, 3), dtype=np.uint8)
        struct3d[1, 1, :] = 1
        struct3d[1, :, 1] = 1
        struct3d[:, 1, 1] = 1
    else:
        struct3d = np.ones((3, 3, 3), dtype=np.uint8)

    labels, n = scipy_label(vol, structure=struct3d)
    return labels, n


# ---------------------------------------------------------------------------
# Colour-map helpers
# ---------------------------------------------------------------------------

def _random_cmap(n_labels: int, seed: int = 42) -> np.ndarray:
    """Return (n_labels+1, 3) uint8 array.  Index 0 = black (background)."""
    rng = np.random.default_rng(seed)
    cmap = rng.integers(40, 256, size=(n_labels + 1, 3), dtype=np.uint8)
    cmap[0] = 0  # background
    return cmap


# ---------------------------------------------------------------------------
# PNG writer (matches codebase convention — no PIL)
# ---------------------------------------------------------------------------

def _write_png(path: Path, img: np.ndarray) -> None:
    """Write an RGB uint8 HxWx3 array as PNG (zlib only)."""
    h, w, _ = img.shape
    raw = b""
    for y in range(h):
        raw += b"\x00" + img[y].tobytes()
    compressed = zlib.compress(raw, 6)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", compressed))
        f.write(_chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Video encoding via ffmpeg
# ---------------------------------------------------------------------------

def _encode_video(
    frames: list[np.ndarray],
    out_path: Path,
    fps: int = 15,
) -> None:
    """Pipe RGB uint8 frames to ffmpeg as mp4."""
    h, w, _ = frames[0].shape
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())  # type: ignore[union-attr]
    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode(errors="replace")  # type: ignore[union-attr]
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}):\n{err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="3D connected components from binary TIFF → uint8 zarr + video",
    )
    p.add_argument("--input", required=True, help="Multi-layer label TIFF (ZYX): 0=bg, 1=pred, 2=ignore")
    p.add_argument("--output", required=True, help="Output CC zarr path (0=bg, 1..N=components)")
    p.add_argument("--binary-zarr", default=None,
                    help="Output binary zarr path (0=bg+ignore, 1=fg)")
    p.add_argument("--video", default=None, help="Output video path (.mp4)")
    p.add_argument("--axis", choices=["z", "y", "x"], default="z",
                    help="Slice axis for video (default: z)")
    p.add_argument("--fps", type=int, default=15, help="Video framerate")
    p.add_argument("--connectivity", type=int, choices=[6, 26], default=6,
                    help="3D connectivity (6=face, 26=full)")
    p.add_argument("--chunk-size", type=int, default=64,
                    help="Zarr chunk size per axis")
    p.add_argument("--min-voxels", type=int, default=0,
                    help="Discard components with fewer voxels (0=keep all)")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # -- read TIFF ----------------------------------------------------------
    print(f"[labels_to_cc] reading {input_path}", flush=True)
    vol = tifffile.imread(str(input_path))
    if vol.ndim == 2:
        vol = vol[np.newaxis]  # single slice → (1, H, W)
    assert vol.ndim == 3, f"expected 3D volume, got shape {vol.shape}"
    print(f"[labels_to_cc] volume shape={vol.shape}  dtype={vol.dtype}", flush=True)

    # Label convention: 0=background, 1=pred/foreground, 2=ignore
    # For CC: only label==1 is foreground; 0 and 2 are both background.
    n_bg = int((vol == 0).sum())
    n_fg = int((vol == 1).sum())
    n_ign = int((vol == 2).sum())
    n_other = vol.size - n_bg - n_fg - n_ign
    print(f"[labels_to_cc] labels: bg(0)={n_bg}  pred(1)={n_fg}  "
          f"ignore(2)={n_ign}  other={n_other}", flush=True)
    if n_other > 0:
        print(f"[labels_to_cc] WARNING: {n_other} voxels with unexpected label values "
              f"(treating as background)", flush=True)

    binary = (vol == 1).astype(np.uint8)
    fg_voxels = int(binary.sum())
    print(f"[labels_to_cc] foreground voxels: {fg_voxels} "
          f"({100*fg_voxels/binary.size:.1f}%)", flush=True)

    # -- connected components -----------------------------------------------
    print(f"[labels_to_cc] running connected components (connectivity={args.connectivity})", flush=True)
    labels, n_raw = _connected_components(binary, args.connectivity)
    print(f"[labels_to_cc] raw components: {n_raw}", flush=True)

    # -- optional size filter -----------------------------------------------
    if args.min_voxels > 0:
        # count voxels per label
        counts = np.bincount(labels.ravel())
        keep = np.where(counts >= args.min_voxels)[0]
        keep = keep[keep > 0]  # exclude background
        # remap
        remap = np.zeros(n_raw + 1, dtype=np.int32)
        for new_id, old_id in enumerate(keep, start=1):
            remap[old_id] = new_id
        labels = remap[labels]
        n_final = int(len(keep))
        print(f"[labels_to_cc] after min_voxels={args.min_voxels}: {n_final} components "
              f"(removed {n_raw - n_final})", flush=True)
    else:
        n_final = n_raw

    if n_final > 254:
        print(f"[labels_to_cc] WARNING: {n_final} components > 254, "
              f"clamping to uint8 range", flush=True)

    labels_u8 = np.clip(labels, 0, 255).astype(np.uint8)

    # -- write zarr ---------------------------------------------------------
    print(f"[labels_to_cc] writing zarr to {output_path}", flush=True)
    cs = args.chunk_size
    arr = zarr.open(
        str(output_path),
        mode="w",
        shape=labels_u8.shape,
        chunks=(cs, cs, cs),
        dtype=np.uint8,
        fill_value=0,
        zarr_format=2,
    )
    arr[:] = labels_u8
    arr.attrs["n_components"] = n_final
    arr.attrs["source"] = str(input_path)
    arr.attrs["connectivity"] = args.connectivity
    arr.attrs["min_voxels"] = args.min_voxels
    print(f"[labels_to_cc] zarr written  shape={arr.shape}  chunks={arr.chunks}", flush=True)

    # -- binary zarr (for vc_gen_normalgrids) --------------------------------
    if args.binary_zarr:
        bin_path = Path(args.binary_zarr)
        print(f"[labels_to_cc] writing binary zarr to {bin_path}", flush=True)
        # vc_gen_normalgrids expects zarr Group with dataset "0" inside
        root = zarr.open_group(str(bin_path), mode="w", zarr_format=2)
        ds = root.create_array(
            "0",
            shape=binary.shape,
            chunks=(cs, cs, cs),
            dtype=np.uint8,
            overwrite=True,
        )
        ds[:] = binary
        print(f"[labels_to_cc] binary zarr written  shape={ds.shape}  chunks={ds.chunks}", flush=True)

    # -- video --------------------------------------------------------------
    if args.video:
        video_path = Path(args.video)
        axis_idx = {"z": 0, "y": 1, "x": 2}[args.axis]
        n_slices = labels_u8.shape[axis_idx]
        cmap = _random_cmap(min(n_final, 255))

        print(f"[labels_to_cc] rendering {n_slices} slices along {args.axis}", flush=True)
        frames: list[np.ndarray] = []
        for i in range(n_slices):
            slc = np.take(labels_u8, i, axis=axis_idx)  # (H, W)
            rgb = cmap[slc]  # (H, W, 3)
            frames.append(rgb)

        # ensure even dimensions for h264
        h, w, _ = frames[0].shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            frames = [
                np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)))
                for f in frames
            ]

        print(f"[labels_to_cc] encoding video to {video_path}  "
              f"({len(frames)} frames, {args.fps} fps)", flush=True)
        _encode_video(frames, video_path, fps=args.fps)
        print(f"[labels_to_cc] video written: {video_path}", flush=True)

    print(f"[labels_to_cc] done — {n_final} components", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
