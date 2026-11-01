"""Check exact rotate/flip pixels in both renderer writers and both render paths.

Run with: python test_render_rotate_flip.py /path/to/vc_render_tifxyz

The volume varies in z, y and x, so a wrong orientation cannot pass by accident. The surface is
curved and sampled at a non-dyadic step, so QuadSurface::gen() output depends in its last bits on
the origin of each call; a rotated render only matches if it builds each tile or band from the same
gen() calls as the unrotated render. Each rotated render is compared with the transformed
unrotated render of the same path (tile path when zarr is written, band path for TIFF only):
every rotation and flip with --quick-tif, a few without it, a crop, and a multi-part rotate 90.
A TIFF-only rotate 90/270 renders source columns in tiles, so it is compared with the tile path.
"""

from pathlib import Path
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest

from render_test_tiff import write_float_tiff


RENDERER = None
Z, VY, VX = 128, 256, 256
VCH, CH = 64, 128
GRID = (121, 81)          # surface grid; at scale 0.4 it renders 302 x 202, not a multiple of 128 or 16
GRID_SCALE = 0.4
SLICES = ["00.tif", "01.tif", "02.tif"]


def make_fixture(root):
    """Create a deep zarr volume and a curved surface that stays inside it."""
    volume = root / "volume.zarr"
    (volume / "0").mkdir(parents=True)
    (volume / ".zgroup").write_text('{"zarr_format":2}\n')
    (volume / ".zattrs").write_text(json.dumps({
        "multiscales": [{
            "version": "0.4",
            "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
            "datasets": [{"path": "0"}],
        }]
    }))
    (volume / "0" / ".zarray").write_text(json.dumps({
        "zarr_format": 2,
        "shape": [Z, VY, VX],
        "chunks": [VCH, CH, CH],
        "dtype": "|u1",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": "/",
    }))
    # value = (29 z + 13 y + 5 x) mod 256; 205 = 1 / 5 mod 256, so each 128-voxel row is one slice of ramp
    ramp = bytes(5 * index & 255 for index in range(512))
    for chunk_z in range(Z // VCH):
        for chunk_y in range(VY // CH):
            for chunk_x in range(VX // CH):
                data = bytearray()
                for z in range(chunk_z * VCH, (chunk_z + 1) * VCH):
                    for y in range(chunk_y * CH, (chunk_y + 1) * CH):
                        start = (z * 29 + y * 13 + chunk_x * CH * 5) * 205 & 255
                        data += ramp[start:start + CH]
                chunk_dir = volume / "0" / str(chunk_z) / str(chunk_y)
                chunk_dir.mkdir(parents=True, exist_ok=True)
                (chunk_dir / str(chunk_x)).write_bytes(data)

    surface = root / "surface"
    surface.mkdir()
    width, height = GRID
    points = [
        (20 + 1.7 * gx + 2.5 * math.sin(gy / 4.3),
         15 + 2.1 * gy + 2.0 * math.sin(gx / 5.1),
         50 + 0.15 * gx + 0.2 * gy + 4.0 * math.sin((gx + gy) / 9.7))
        for gy in range(height)
        for gx in range(width)
    ]
    for axis, name in enumerate("xyz"):
        write_float_tiff(surface / f"{name}.tif", width, height, [p[axis] for p in points])
    low = [min(p[axis] for p in points) for axis in range(3)]
    high = [max(p[axis] for p in points) for axis in range(3)]
    assert all(lo > 1 for lo in low) and high[0] < VX - 2 and high[1] < VY - 2 and high[2] < Z - 2
    (surface / "meta.json").write_text(json.dumps({
        "format": "tifxyz",
        "type": "seg",
        "uuid": "rotate-flip",
        "scale": [GRID_SCALE, GRID_SCALE],
        "bbox": [low, high],
    }))
    return volume, surface


TIFF_TAGS = {
    256: "width",
    257: "height",
    259: "compression",
    317: "predictor",
    322: "tile_width",
    323: "tile_height",
    324: "offsets",
    325: "counts",
}
TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}
TYPE_FORMAT = {1: "B", 3: "H", 4: "I"}


def tiff_tags(path):
    raw = Path(path).read_bytes()
    byte_order = "<" if raw[:2] == b"II" else ">"
    _, offset = struct.unpack(byte_order + "HI", raw[2:8])
    count = struct.unpack(byte_order + "H", raw[offset:offset + 2])[0]
    page = {}
    for index in range(count):
        entry = offset + 2 + 12 * index
        tag, kind, values = struct.unpack(byte_order + "HHI", raw[entry:entry + 8])
        size = TYPE_SIZE.get(kind, 4) * values
        if size <= 4:
            data = raw[entry + 8:entry + 8 + size]
        else:
            data_offset = struct.unpack(byte_order + "I", raw[entry + 8:entry + 12])[0]
            data = raw[data_offset:data_offset + size]
        if kind in TYPE_FORMAT and len(data) >= size:
            page[TIFF_TAGS.get(tag, tag)] = list(struct.unpack(byte_order + f"{values}{TYPE_FORMAT[kind]}", data))
    return raw, page


def unpack_packbits(data):
    output = bytearray()
    index = 0
    while index < len(data):
        count = struct.unpack("b", data[index:index + 1])[0]
        index += 1
        if 0 <= count <= 127:
            output += data[index:index + count + 1]
            index += count + 1
        elif -127 <= count <= -1:
            output += data[index:index + 1] * (-count + 1)
            index += 1
    return bytes(output)


def unpack_lzw(data):
    """TIFF LZW: MSB-first codes of 9 to 12 bits, code width grows one code early."""
    output = bytearray()
    table = [bytes([value]) for value in range(256)] + [b"", b""]
    width, previous, bits, nbits = 9, None, 0, 0
    for byte in data:
        bits = (bits << 8) | byte
        nbits += 8
        while nbits >= width:
            nbits -= width
            code = (bits >> nbits) & ((1 << width) - 1)
            if code == 256:
                del table[258:]
                width, previous = 9, None
                continue
            if code == 257:
                return bytes(output)
            if previous is None:
                entry = table[code]
            else:
                entry = table[code] if code < len(table) else previous + previous[:1]
                table.append(previous + entry[:1])
                if len(table) + 1 >= (1 << width) and width < 12:
                    width += 1
            output += entry
            previous = entry
    return bytes(output)


def read_tiff(path):
    raw, tags = tiff_tags(path)
    width = tags["width"][0]
    height = tags["height"][0]
    tile_width = tags["tile_width"][0]
    tile_height = tags["tile_height"][0]
    compression = tags["compression"][0]
    pixels = [0] * (width * height)
    tiles_x = (width + tile_width - 1) // tile_width
    for tile_index, (offset, count) in enumerate(zip(tags["offsets"], tags["counts"])):
        compressed = raw[offset:offset + count]
        if compression == 5:
            tile = bytearray(unpack_lzw(compressed))
            if tags.get("predictor", [1])[0] == 2:
                for start in range(0, len(tile), tile_width):
                    for column in range(start + 1, start + tile_width):
                        tile[column] = (tile[column] + tile[column - 1]) & 255
        elif compression == 32773:
            tile = unpack_packbits(compressed)
        else:
            raise AssertionError(f"unexpected TIFF compression {compression}")
        tile_x = tile_index % tiles_x
        tile_y = tile_index // tiles_x
        for row in range(min(tile_height, height - tile_y * tile_height)):
            source = tile[row * tile_width:(row + 1) * tile_width]
            destination = (tile_y * tile_height + row) * width + tile_x * tile_width
            length = min(tile_width, width - tile_x * tile_width)
            pixels[destination:destination + length] = source[:length]
    return width, height, pixels


def read_zarr(root):
    """Read output chunks, including all planes stored in each z chunk."""
    metadata = json.loads((root / "0" / ".zarray").read_text())
    depth, height, width = metadata["shape"]
    chunk_z, chunk_y, chunk_x = metadata["chunks"]
    planes = []
    for z in range(depth):
        plane = [0] * (height * width)
        chunk_z_index, local_z = divmod(z, chunk_z)
        for chunk_y_index in range((height + chunk_y - 1) // chunk_y):
            for chunk_x_index in range((width + chunk_x - 1) // chunk_x):
                path = root / "0" / str(chunk_z_index) / str(chunk_y_index) / str(chunk_x_index)
                if not path.exists():
                    continue
                data = path.read_bytes()
                for row in range(min(chunk_y, height - chunk_y_index * chunk_y)):
                    length = min(chunk_x, width - chunk_x_index * chunk_x)
                    source = (local_z * chunk_y + row) * chunk_x
                    destination = (chunk_y_index * chunk_y + row) * width + chunk_x_index * chunk_x
                    plane[destination:destination + length] = data[source:source + length]
        planes.append((width, height, plane))
    return planes


def orient(image, angle, flip):
    width, height, pixels = image
    rows = [pixels[row * width:(row + 1) * width] for row in range(height)]
    if angle == 90:
        rows = [list(column) for column in zip(*rows)][::-1]
    elif angle == 180:
        rows = [row[::-1] for row in rows[::-1]]
    elif angle == 270:
        rows = [list(column) for column in zip(*rows[::-1])]
    if flip in (0, 2):
        rows = rows[::-1]
    if flip in (1, 2):
        rows = [row[::-1] for row in rows]
    return len(rows[0]), len(rows), [value for row in rows for value in row]


class RotateFlipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="vc render rotate flip ")
        cls.root = Path(cls.temporary.name)
        cls.volume, cls.surface = make_fixture(cls.root)
        cls.references = {}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_render(self, output, angle=None, flip=None, zarr=True, quick=True, pyramid=True, extra=()):
        args = [
            RENDERER,
            "-v", str(self.volume),
            "-s", str(self.surface),
            "--scale", "1",
            "-g", "0",
            "--num-slices", "3",
            "--slice-step", "1",
            "--tif-output", str(output / "tif"),
        ]
        if quick:
            args.append("--quick-tif")
        if zarr:
            args += ["--zarr-output", str(output / "zarr"), "--zarr-compressor", "none"]
            if not pyramid:
                args += ["--pyramid", "false"]
        if angle is not None:
            args += ["--rotate", str(angle)]
        if flip is not None:
            args += ["--flip", str(flip)]
        result = subprocess.run(
            args + list(extra),
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ, HOME=str(self.root), USERPROFILE=str(self.root)),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def read_tiffs(self, output):
        self.assertEqual(sorted(path.name for path in (output / "tif").iterdir()), SLICES, output)
        return [read_tiff(output / "tif" / name) for name in SLICES]

    def reference(self, tile_path, quick, crop=()):
        """The unrotated render of one path: {"tif": [...], "zarr": [...] or None}."""
        key = (tile_path, quick, crop)
        if key not in self.references:
            output = self.root / f"ref-{'tile' if tile_path else 'band'}-{int(quick)}-{len(crop)}"
            self.run_render(output, zarr=tile_path, quick=quick, pyramid=False, extra=crop)
            self.references[key] = {
                "tif": self.read_tiffs(output),
                "zarr": read_zarr(output / "zarr") if tile_path else None,
            }
        return self.references[key]

    def assert_image(self, got, expected, label):
        """Like assertEqual, without unittest's slow diff of two long pixel lists."""
        self.assertEqual(got[:2], expected[:2], label)
        self.assertEqual(len(got[2]), got[0] * got[1], label)
        self.assertEqual(len(expected[2]), expected[0] * expected[1], label)
        diff = [index for index, (a, b) in enumerate(zip(got[2], expected[2])) if a != b]
        if diff:
            y, x = divmod(diff[0], got[0])
            self.fail(f"{label}: {len(diff)} of {len(got[2])} pixels differ, first at x={x} y={y}: "
                      f"{got[2][diff[0]]} != {expected[2][diff[0]]}")

    def check(self, output, reference, angle, flip, zarr=True):
        for z, (got, image) in enumerate(zip(self.read_tiffs(output), reference["tif"])):
            self.assert_image(got, orient(image, angle, flip), ("tif", z, output))
        if zarr:
            got = read_zarr(output / "zarr")
            self.assertEqual(len(got), len(reference["zarr"]))
            for z, (plane, image) in enumerate(zip(got, reference["zarr"])):
                self.assert_image(plane, orient(image, angle, flip), ("zarr", z, output))

    def check_variant(self, angle, flip, quick=True, crop=()):
        tag = f"r{angle}f{flip}q{int(quick)}c{len(crop)}"
        with self.subTest(angle=angle, flip=flip, quick=quick, crop=bool(crop), path="tile"):
            output = self.root / f"tile-{tag}"
            self.run_render(output, angle, flip, quick=quick, pyramid=False, extra=crop)
            self.check(output, self.reference(True, quick, crop), angle, flip)
        with self.subTest(angle=angle, flip=flip, quick=quick, crop=bool(crop), path="tif-only"):
            output = self.root / f"band-{tag}"
            self.run_render(output, angle, flip, zarr=False, quick=quick, pyramid=False, extra=crop)
            self.check(output, self.reference(angle in (90, 270), quick, crop), angle, flip, zarr=False)

    def test_fixture_varies(self):
        reference = self.reference(True, False)
        width, height, pixels = reference["tif"][0]
        self.assertEqual((width, height), (302, 202))
        self.assertEqual(len({tuple(image[2]) for image in reference["tif"]}), 3)
        rows = {tuple(pixels[row * width:(row + 1) * width]) for row in range(height)}
        self.assertGreater(len(rows), 1)
        for z, (image, plane) in enumerate(zip(reference["tif"], reference["zarr"])):
            self.assert_image(image, plane, ("tif vs zarr", z))

    def test_all_orientations_quick_tif(self):
        for angle in (0, 90, 180, 270):
            for flip in (-1, 0, 1, 2):
                if (angle, flip) != (0, -1):
                    self.check_variant(angle, flip)

    def test_exact_without_quick_tif(self):
        for angle, flip in ((90, -1), (180, -1), (0, 0)):
            self.check_variant(angle, flip, quick=False)

    def test_crop(self):
        crop = ("--crop-x", "37", "--crop-y", "21", "--crop-width", "211", "--crop-height", "150")
        self.check_variant(270, -1, quick=False, crop=crop)
        self.check_variant(180, -1, quick=False, crop=crop)

    def test_multi_part_merge(self):
        output = self.root / "parts"
        self.run_render(output, 90, extra=("--pre",))
        for part in (0, 1):
            self.run_render(output, 90, extra=("--num-parts", "2", "--part-id", str(part)))
        result = subprocess.run(
            [
                RENDERER,
                "-g", "0",
                "--scale", "1",
                "--merge-tiff-parts",
                "--tif-output", str(output / "tif"),
                "--num-parts", "2",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check(output, self.reference(True, True), 90, -1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: test_render_rotate_flip.py /path/to/vc_render_tifxyz", file=sys.stderr)
        sys.exit(2)
    RENDERER = str(Path(sys.argv.pop(1)).resolve())
    unittest.main()
