"""Render/write/resume regression, using only the Python standard library.

Run: python test_render_zarr_output.py /path/to/vc_render_tifxyz
"""

import array
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from render_test_tiff import write_float_tiff

RENDERER = str(Path(sys.argv.pop(1)).resolve())


def read_raw(path):
    meta = json.loads((path / ".zarray").read_text())
    shape, chunks = meta["shape"], meta["chunks"]
    dtype = "H" if meta["dtype"] == "<u2" else "B"
    out = array.array(dtype, [0]) * (shape[0] * shape[1] * shape[2])
    grid = [(s + c - 1) // c for s, c in zip(shape, chunks)]
    for idx in itertools.product(*(range(n) for n in grid)):
        p = path / meta.get("dimension_separator", ".").join(map(str, idx))
        if not p.exists():
            continue
        data = array.array(dtype, p.read_bytes())
        if sys.byteorder != "little":
            data.byteswap()
        assert len(data) == chunks[0] * chunks[1] * chunks[2]
        for z in range(min(chunks[0], shape[0] - idx[0] * chunks[0])):
            for y in range(min(chunks[1], shape[1] - idx[1] * chunks[1])):
                n = min(chunks[2], shape[2] - idx[2] * chunks[2])
                src = (z * chunks[1] + y) * chunks[2]
                dst = ((idx[0] * chunks[0] + z) * shape[1] + idx[1] * chunks[1] + y) * shape[2] + idx[2] * chunks[2]
                out[dst:dst + n] = data[src:src + n]
    return shape, out


class RenderZarrTests(unittest.TestCase):
    def test_write_pyramid_resume_and_precreated_parts(self):
        for dtype in ("|u1", "<u2"):
            for separator in ("/", "."):
                with self.subTest(dtype=dtype, separator=separator), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    volume = root / "volume" / "0"
                    volume.mkdir(parents=True)
                    (volume.parent / ".zgroup").write_text('{"zarr_format":2}')
                    (volume / ".zarray").write_text(json.dumps({
                        "zarr_format": 2, "shape": [16, 272, 272], "chunks": [16, 272, 272],
                        "dtype": dtype, "compressor": None, "fill_value": 0, "filters": None, "order": "C",
                    }))
                    # Independent of z so the normal's direction does not change the reference.
                    values = array.array("H" if dtype == "<u2" else "B", (
                        (1000 if dtype == "<u2" else 10) + (y + x) % 101
                        for z in range(16) for y in range(272) for x in range(272)))
                    if sys.byteorder != "little":
                        values.byteswap()
                    (volume / "0.0.0").write_bytes(values.tobytes())
                    surface = root / "surface"
                    surface.mkdir()
                    width, height = 257, 259
                    for name, values in (
                        ("x", [float(x + 4) for y in range(height) for x in range(width)]),
                        ("y", [float(y + 4) for y in range(height) for x in range(width)]),
                        ("z", [8.0] * (width * height)),
                    ):
                        write_float_tiff(surface / f"{name}.tif", width, height, values)
                    (surface / "meta.json").write_text(json.dumps({
                        "format": "tifxyz", "type": "seg", "uuid": "surface", "scale": [1, 1],
                        "bbox": [[4, 4, 8], [width + 3, height + 3, 8]],
                    }))
                    args = [RENDERER, "-v", str(volume.parent), "-s", str(surface), "-g", "0",
                            "--scale", "1", "--num-slices", "3", "--zarr-compressor", "none",
                            "--zarr-separator", separator, "--pyramid", "true"]

                    def render(out, extra=(), pyramid=True):
                        command = args.copy()
                        command[command.index("--pyramid") + 1] = str(pyramid).lower()
                        result = subprocess.run(command + ["--zarr-output", str(out)] + list(extra),
                                                capture_output=True, text=True, timeout=60,
                                                env=dict(os.environ, OMP_NUM_THREADS="2"))
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

                    out = root / "render"
                    render(out)
                    reference = [read_raw(out / str(level)) for level in range(6)]
                    shape, pixels = reference[0]
                    self.assertEqual(json.loads((out / "0" / ".zarray").read_text())["dtype"], dtype)
                    self.assertEqual(shape, [3, height, width])
                    self.assertGreater(max(pixels), 0)
                    # Resume reads stored L0 chunks to reconstruct the inline pyramid.
                    for level in range(1, 6):
                        for p in (out / str(level)).rglob("*"):
                            if p.is_file() and not p.name.startswith("."):
                                p.unlink()
                    render(out, ["--resume"])
                    self.assertEqual([read_raw(out / str(level)) for level in range(6)], reference)
                    # --pre creates arrays; separate workers reopen them for writes.
                    parts = root / "parts"
                    render(parts, ["--pre"])
                    for part in range(2):
                        render(parts, ["--num-parts", "2", "--part-id", str(part)], pyramid=False)
                    self.assertEqual(read_raw(parts / "0"), reference[0])
                    # Rotation forces the on-disk pyramid builder instead of inline accumulation.
                    aligned = root / "aligned"
                    crop = ["--crop-width", "256", "--crop-height", "256"]
                    render(aligned, crop)
                    aligned_shape, aligned_pixels = read_raw(aligned / "0")
                    rotated = root / "rotated"
                    render(rotated, crop + ["--rotate", "180"])
                    rot_shape, rot_pixels = read_raw(rotated / "0")
                    self.assertEqual(rot_shape, aligned_shape)
                    plane = 256 * 256
                    for z in range(3):
                        self.assertEqual(rot_pixels[z * plane:(z + 1) * plane],
                                         aligned_pixels[z * plane:(z + 1) * plane][::-1])
                    self.assertGreater(max(read_raw(rotated / "5")[1]), 0)



if __name__ == "__main__":
    unittest.main()
