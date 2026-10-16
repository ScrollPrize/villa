"""CLI regression: a --gpu render writes the bytes the CPU render writes.

Run with: python test_render_tifxyz_gpu.py /path/to/vc_render_tifxyz

--gpu moves only the volume sampling (apps/src/RenderCuda.cpp) to a CUDA device; surface generation,
accumulation, rotation and the writers stay on the CPU, so the output of a command line must not depend
on the flag. The volume is a small synthetic local OME-Zarr (uncompressed, an odd chunk shape, two chunks
missing), the surface a curved tifxyz sheet with holes that leaves the volume at its edges, so samples
fall outside the volume, in missing chunks and across chunk borders. Each case renders the same command
line with and without --gpu and compares the chunk files of the two zarr outputs (or the TIFFs) byte for
byte, for uint8 and uint16 sources, fractional steps with accumulation, the composite reducers, a scaled
and cropped canvas and the TIFF band path.

A machine whose renderer reports "GPU unavailable" skips every case. A renderer that fell back to the
CPU for another reason fails, because the GPU summary line is required in its output.

Standard library only, like the other vc_render_tifxyz tests. Under a minute.
"""

from pathlib import Path
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest

from render_test_tiff import write_float_tiff

if len(sys.argv) < 2:
    raise SystemExit("usage: test_render_tifxyz_gpu.py /path/to/vc_render_tifxyz")
RENDERER = str(Path(sys.argv.pop(1)).resolve())

# Volume extent and chunk shape in voxels: neither cubic nor a divisor of the extent, so the last
# chunk of every axis is partial, and the sheet below leaves the volume on three sides.
Z, Y, X = 96, 80, 112
CZ, CY, CX = 32, 24, 40
MISSING = {(1, 1, 1), (2, 0, 2)}  # not stored: reads as 0
SHEET_W, SHEET_H = 150, 137


def write_volume(root, name, dtype):
    """An uncompressed zarr v2 pyramid of one level with pseudo-random voxels."""
    vol = root / name
    (vol / "0").mkdir(parents=True)
    (vol / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (vol / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "version": "0.4",
                        "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
                        "datasets": [{"path": "0"}],
                    }
                ]
            }
        )
    )
    (vol / "0" / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [Z, Y, X],
                "chunks": [CZ, CY, CX],
                "dtype": dtype,
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
            }
        )
    )
    rng = random.Random(1234 if dtype == "|u1" else 4321)
    n = CZ * CY * CX * (1 if dtype == "|u1" else 2)
    for cz in range((Z + CZ - 1) // CZ):
        for cy in range((Y + CY - 1) // CY):
            for cx in range((X + CX - 1) // CX):
                if (cz, cy, cx) in MISSING:
                    continue
                (vol / "0" / ("%d.%d.%d" % (cz, cy, cx))).write_bytes(rng.randbytes(n))
    return vol


def write_sheet(root):
    """A curved tifxyz sheet, one grid step per ~0.8 voxel, with a few holes (-1 nodes)."""
    d = root / "sheet"
    d.mkdir()
    xs, ys, zs = [], [], []
    valid = []
    for r in range(SHEET_H):
        for c in range(SHEET_W):
            x = 3.0 + 0.78 * c + 2.5 * math.sin(0.21 * r)
            y = 8.0 + 0.55 * r + 0.13 * c
            z = 44.0 + 7.0 * math.sin(0.11 * c) + 0.37 * r
            if (20 <= r < 24 and 30 <= c < 35) or (r == 50 and c == 60) or (c == 99 and r % 7 == 0):
                x = y = z = -1.0
            else:
                valid.append((x, y, z))
            xs.append(x)
            ys.append(y)
            zs.append(z)
    for name, vals in (("x", xs), ("y", ys), ("z", zs)):
        write_float_tiff(d / ("%s.tif" % name), SHEET_W, SHEET_H, vals)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "format": "tifxyz",
                "type": "seg",
                "uuid": "sheet",
                "scale": [1.0, 1.0],
                "bbox": [[min(v[i] for v in valid) for i in range(3)], [max(v[i] for v in valid) for i in range(3)]],
            }
        )
    )
    return d


def output_files(path):
    """Every file under path (the zarr chunks or the TIFFs) by its relative name, without the
    .zattrs, which describes the run rather than the pixels."""
    return {
        p.relative_to(path).as_posix(): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != ".zattrs"
    }


class GpuRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="vc render gpu ")
        cls.root = Path(cls.temp.name)
        cls.u8 = write_volume(cls.root, "u8.zarr", "|u1")
        cls.u16 = write_volume(cls.root, "u16.zarr", "<u2")
        cls.sheet = write_sheet(cls.root)
        # One probe render decides whether this machine can run the GPU cases at all.
        out, text = cls.render_to(cls.root / "probe", cls.u8, ["--num-slices", "1"], gpu=True, tif=True)
        if "GPU unavailable" in text:
            cls.temp.cleanup()
            reason = [line for line in text.splitlines() if "GPU unavailable" in line][0]
            raise unittest.SkipTest(reason.strip())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def render_to(cls, out, volume, extra, gpu, tif=False, scale="1"):
        args = [RENDERER, "-v", str(volume), "-s", str(cls.sheet), "--scale", scale, "-g", "0",
                "--cache-gb", "1", "--timeout", "5"]
        if tif:
            args += ["--tif-output", str(out)]
        else:
            args += ["--zarr-output", str(out), "--zarr-compressor", "none", "--pyramid", "0"]
        args += list(extra)
        if gpu:
            args.append("--gpu")
        r = subprocess.run(args, capture_output=True, text=True, timeout=300)
        text = r.stdout + r.stderr
        if r.returncode != 0:
            raise AssertionError("render failed (%s): %s" % (" ".join(args), text[-1200:]))
        return out, text

    def check(self, tag, volume, extra, tif=False, scale="1"):
        cpu, _ = self.render_to(self.root / (tag + "_cpu"), volume, extra, gpu=False, tif=tif, scale=scale)
        gpu, text = self.render_to(self.root / (tag + "_gpu"), volume, extra, gpu=True, tif=tif, scale=scale)
        self.assertNotIn("rendering on the CPU", text, "the GPU render fell back to the CPU:\n" + text[-800:])
        self.assertRegex(text, r"GPU: \d+ bands in \d+ pass", "no GPU summary line in:\n" + text[-800:])
        a, b = output_files(cpu), output_files(gpu)
        self.assertEqual(sorted(a), sorted(b), "the two renders wrote different files")
        self.assertTrue(any(any(v) for k, v in a.items() if not k.endswith((".zarray", ".zgroup"))),
                        "the CPU render is empty, so the comparison proves nothing")
        for name in sorted(a):
            if a[name] == b[name]:
                continue
            first = next(i for i, (p, q) in enumerate(zip(a[name], b[name])) if p != q) \
                if len(a[name]) == len(b[name]) else -1
            self.fail("%s: --gpu differs from the CPU render (sizes %d and %d, first difference at byte %d)"
                      % (name, len(a[name]), len(b[name]), first))
        return text

    def test_uint8_layers(self):
        self.check("u8", self.u8, ["--num-slices", "21", "--slice-step", "1", "--flip-normals"])

    def test_uint8_fractional_step_with_accumulation(self):
        # sub-offsets are sampled on the device, the mean is taken on the CPU
        self.check("u8_accum", self.u8, ["--num-slices", "5", "--slice-step", "1.5", "--accum", "0.5",
                                         "--accum-type", "mean"])

    def test_uint16_layers(self):
        # uint16 samples round (uint8 ones truncate) and clamp at 65535
        self.check("u16", self.u16, ["--num-slices", "7", "--slice-step", "0.8"])

    def test_composite_max(self):
        self.check("cmax", self.u8, ["--composite-collapse", "--accum-type", "max",
                                     "--composite-start=-6", "--composite-end=6"])

    def test_composite_median_with_cutoff(self):
        self.check("cmed", self.u8, ["--composite-collapse", "--accum-type", "median",
                                     "--composite-start=-3", "--composite-end=4", "--iso-cutoff", "40"])

    def test_composite_mean_scaled_and_cropped(self):
        # the tile geometry of a scaled, cropped canvas is generated per tile and gathered per row
        self.check("cmean", self.u8, ["--composite-collapse", "--accum-type", "mean",
                                      "--composite-start=-2", "--composite-end=2",
                                      "--crop-x", "40", "--crop-y", "20", "--crop-width", "150",
                                      "--crop-height", "110"], scale="2")

    def test_tiff_bands(self):
        # the TIFF-only path samples whole bands (renderBands) rather than tile rows
        self.check("tif", self.u8, ["--num-slices", "3", "--slice-step", "1"], tif=True)

    def test_alpha_composite_stays_on_the_cpu(self):
        extra = ["--composite-collapse", "--accum-type", "alpha", "--composite-start=-2", "--composite-end=2"]
        cpu, _ = self.render_to(self.root / "alpha_cpu", self.u8, extra, gpu=False)
        gpu, text = self.render_to(self.root / "alpha_gpu", self.u8, extra, gpu=True)
        self.assertIn("is not available on the GPU; rendering on the CPU", text)
        self.assertNotRegex(text, r"GPU: \d+ bands")
        self.assertEqual(output_files(cpu), output_files(gpu))


if __name__ == "__main__":
    unittest.main(verbosity=2)
