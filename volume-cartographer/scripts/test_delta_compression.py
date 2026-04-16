#!/usr/bin/env python3
"""
Test delta compression between adjacent z-slices in a zarr volume.

Adjacent z-slices in volumetric CT data are very similar. This script measures
whether encoding the delta (slice[z] - slice[z-1]) gives better compression
ratios than encoding the raw slices directly.

Usage:
    python test_delta_compression.py /path/to/volume.zarr
    python test_delta_compression.py https://example.com/volume.zarr
    python test_delta_compression.py /path/to/volume.zarr --chunks 5
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np


def fetch_bytes(url_or_path: str) -> bytes:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        req = urllib.request.Request(url_or_path)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    return Path(url_or_path).read_bytes()


def fetch_json(url_or_path: str) -> dict:
    return json.loads(fetch_bytes(url_or_path))


def read_chunk(base_url: str, zarray: dict, cz: int, cy: int, cx: int) -> np.ndarray:
    """Read and decompress a single zarr chunk, returning a 3D uint8 array."""
    sep = zarray.get("dimension_separator", ".")
    if sep == "/":
        chunk_path = f"{cz}/{cy}/{cx}"
    else:
        chunk_path = f"{cz}.{cy}.{cx}"

    raw = fetch_bytes(f"{base_url}/{chunk_path}")

    compressor = zarray.get("compressor")
    chunks = zarray["chunks"]
    expected = chunks[0] * chunks[1] * chunks[2]

    if compressor is None:
        data = np.frombuffer(raw, dtype=np.uint8)
    elif compressor.get("id") == "blosc":
        import blosc
        data = np.frombuffer(blosc.decompress(raw), dtype=np.uint8)
    elif compressor.get("id") == "blosc2":
        import blosc2
        data = np.frombuffer(blosc2.decompress(raw), dtype=np.uint8)
    elif compressor.get("id") == "zlib":
        import zlib
        data = np.frombuffer(zlib.decompress(raw), dtype=np.uint8)
    else:
        # Try raw — might be VC3D or uncompressed
        data = np.frombuffer(raw, dtype=np.uint8)

    if data.size < expected:
        raise RuntimeError(f"chunk too small: {data.size} < {expected}")
    return data[:expected].reshape(chunks)


def encode_h265(volume: np.ndarray, qp: int) -> bytes:
    """Encode a (Z, Y, X) uint8 volume with H.265 via ffmpeg. Returns raw bitstream."""
    z, y, x = volume.shape
    pad_y = (y + 1) & ~1
    pad_x = (x + 1) & ~1
    if pad_y != y or pad_x != x:
        padded = np.zeros((z, pad_y, pad_x), dtype=np.uint8)
        padded[:, :y, :x] = volume
        volume = padded

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{pad_x}x{pad_y}",
        "-r", "1",
        "-i", "pipe:0",
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-x265-params", f"qp={qp}:log-level=none",
        "-f", "hevc",
        "pipe:1",
    ]

    proc = subprocess.run(cmd, input=volume.tobytes(), capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg h265 encode failed: {proc.stderr.decode()}")
    return proc.stdout


def compute_delta(volume: np.ndarray) -> np.ndarray:
    """Compute delta encoding: delta[0] = slice[0], delta[z] = slice[z] - slice[z-1] (wrapping uint8)."""
    delta = np.empty_like(volume)
    delta[0] = volume[0]
    delta[1:] = np.diff(volume.astype(np.uint8), axis=0).astype(np.uint8)
    return delta


def verify_delta_roundtrip(volume: np.ndarray, delta: np.ndarray):
    """Verify that delta encoding is perfectly reversible."""
    reconstructed = np.empty_like(delta)
    reconstructed[0] = delta[0]
    for z in range(1, delta.shape[0]):
        reconstructed[z] = (reconstructed[z - 1].astype(np.uint16) + delta[z].astype(np.uint16)).astype(np.uint8)
    assert np.array_equal(volume, reconstructed), "delta roundtrip failed!"


def test_chunk(volume: np.ndarray, label: str):
    """Test original vs delta encoding at QP15 and QP51."""
    raw_size = volume.nbytes
    delta = compute_delta(volume)
    verify_delta_roundtrip(volume, delta)

    # Measure delta statistics
    delta_nonzero = np.count_nonzero(delta[1:])
    delta_total = delta[1:].size
    delta_zero_pct = 100.0 * (1.0 - delta_nonzero / delta_total)
    delta_mean = np.mean(np.abs(delta[1:].astype(np.int8).astype(np.float32)))

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  shape: {volume.shape}, raw size: {raw_size:,} bytes")
    print(f"  delta stats: {delta_zero_pct:.1f}% zero, mean |delta|: {delta_mean:.1f}")
    print(f"{'='*70}")
    print(f"  {'Mode':<12} {'QP':>4} {'Encoded':>12} {'Ratio':>8} {'Savings':>10}")
    print(f"  {'-'*50}")

    results = []
    for qp in [15, 51]:
        orig_enc = encode_h265(volume, qp)
        delta_enc = encode_h265(delta, qp)
        orig_ratio = raw_size / len(orig_enc)
        delta_ratio = raw_size / len(delta_enc)
        savings = 100.0 * (1.0 - len(delta_enc) / len(orig_enc))

        print(f"  {'original':<12} {qp:>4} {len(orig_enc):>10,} B {orig_ratio:>7.1f}x")
        print(f"  {'delta':<12} {qp:>4} {len(delta_enc):>10,} B {delta_ratio:>7.1f}x {savings:>+8.1f}%")

        results.append({
            "qp": qp,
            "orig_bytes": len(orig_enc),
            "delta_bytes": len(delta_enc),
            "orig_ratio": orig_ratio,
            "delta_ratio": delta_ratio,
            "savings_pct": savings,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test delta compression between adjacent z-slices in a zarr volume."
    )
    parser.add_argument("volume", help="Path or URL to an OME-Zarr volume")
    parser.add_argument("--level", type=int, default=0, help="Pyramid level (default: 0)")
    parser.add_argument("--chunks", type=int, default=3, help="Number of chunks to test (default: 3)")
    args = parser.parse_args()

    base = args.volume.rstrip("/")
    level_url = f"{base}/{args.level}"

    print(f"Reading zarr metadata from {level_url}/.zarray")
    zarray = fetch_json(f"{level_url}/.zarray")
    shape = zarray["shape"]
    chunks = zarray["chunks"]
    print(f"Volume shape: {shape}, chunk size: {chunks}")

    if chunks[0] < 2:
        print("ERROR: z chunk size must be >= 2 for delta encoding test", file=sys.stderr)
        return 1

    # Pick chunks spread across the volume
    nz = (shape[0] + chunks[0] - 1) // chunks[0]
    ny = (shape[1] + chunks[1] - 1) // chunks[1]
    nx = (shape[2] + chunks[2] - 1) // chunks[2]

    # Sample from the middle of the volume for representative data
    mid_z = nz // 2
    mid_y = ny // 2
    mid_x = nx // 2

    coords = []
    for i in range(args.chunks):
        cz = min(mid_z + i, nz - 1)
        coords.append((cz, mid_y, mid_x))

    all_results = []
    for cz, cy, cx in coords:
        label = f"chunk [{cz},{cy},{cx}]"
        print(f"\nFetching {label}...")
        try:
            volume = read_chunk(level_url, zarray, cz, cy, cx)
        except Exception as e:
            print(f"  skipping: {e}")
            continue
        results = test_chunk(volume, label)
        all_results.append(results)

    if not all_results:
        print("\nNo chunks could be read.", file=sys.stderr)
        return 1

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for qp in [15, 51]:
        savings = [r["savings_pct"] for chunk_results in all_results for r in chunk_results if r["qp"] == qp]
        if savings:
            avg = sum(savings) / len(savings)
            print(f"  QP {qp}: avg delta savings = {avg:+.1f}% across {len(savings)} chunks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
