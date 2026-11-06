"""The layers of vc_render_tifxyz bend smoothly across grid cells. Run with: python test_render_tifxyz_normals_min.py /path/to/vc_render_tifxyz"""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np, tifffile
RENDERER = str(Path(sys.argv.pop(1)).resolve())
STEP, N, R = 8, 11, 150.0

class NormalsTest(unittest.TestCase):
    def test_layer_has_no_kink_at_cell_edges(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); mesh = t / "m.tifxyz"; mesh.mkdir()
            y, z = [(a - (N - 1) / 2) * STEP for a in np.meshgrid(np.arange(N), np.arange(N), indexing="ij")]
            for name, v in (("x", np.sqrt(R * R - y * y - z * z)), ("y", y + 60), ("z", z + 60)):
                tifffile.imwrite(mesh / f"{name}.tif", v.astype(np.float32))
            (mesh / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "uuid": "s", "scale": [1 / STEP, 1 / STEP]}))
            vol = t / "v.zarr"; (vol / "0").mkdir(parents=True); (vol / ".zgroup").write_text('{"zarr_format": 2}')
            shape = (130, 130, 260)  # the value is 2 * (z + 60): it reads the z coordinate of the layer
            (vol / "0" / ".zarray").write_text(json.dumps({"zarr_format": 2, "shape": list(shape), "chunks": list(shape), "dtype": "|u1", "compressor": None, "filters": None, "fill_value": 0, "order": "C", "dimension_separator": "."}))
            (vol / "0" / "0.0.0").write_bytes(np.broadcast_to((2 * np.arange(130))[:, None, None], shape).astype(np.uint8).tobytes())
            subprocess.run([RENDERER, "-v", str(vol), "-s", str(mesh), "-g", "0", "--scale", "1", "--num-slices", "3", "--slice-step", "60", "--surface-interpolation", "linear", "--pyramid", "0", "--tif-output", str(t / "o")], check=True, capture_output=True)
            layer = tifffile.imread(t / "o" / "00.tif").astype(float)[20:-20, 20:-20]
            kink = np.abs(np.diff(layer, 2, axis=1).mean(axis=0)).max()  # mean over rows of the second difference along the columns
        self.assertLess(kink, 3.0, "kink of the offset layer at a cell edge, grey levels")

if __name__ == "__main__":
    unittest.main()
