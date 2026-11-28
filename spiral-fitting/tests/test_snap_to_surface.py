import json
import sys
import unittest
from pathlib import Path

import numpy as np

SPIRAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIRAL_DIR))

import snap_to_surface  # noqa: E402

CENTRE = 80.0
SHEETS = (40.0, 50.0, 60.0, 70.0)


def sheet_volume(depth=32, size=160):
    """uint8 volume with concentric cylindrical sheets (2 voxels thick) around the centre."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    r = np.hypot(yy - CENTRE, xx - CENTRE)
    plane = np.zeros((size, size), np.uint8)
    for radius in SHEETS:
        plane[np.abs(r - radius) < 1.0] = 255
    return np.repeat(plane[None], depth, 0)


class ChunkedArray(np.ndarray):
    """A numpy volume with the zarr `chunks` attribute that snap() reads."""
    chunks = (8, 160, 160)


def cylinder(radius, depth=32, cols=181):
    """A winding grid (rows = z, cols = angle) as x, y, z."""
    th = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    z = np.arange(depth, dtype=np.float32)
    zz, tt = np.meshgrid(z, th, indexing='ij')
    return np.stack([CENTRE + radius * np.cos(tt), CENTRE + radius * np.sin(tt), zz], -1).astype(np.float32)


def run(radii):
    S = sheet_volume().view(ChunkedArray)
    windings = {w: (f'w{w:03d}', cylinder(r), {}) for w, r in enumerate(radii, start=1)}
    res, _ = snap_to_surface.snap(windings, S, lambda z: (CENTRE, CENTRE), (2, 30), rays=256, rmax=100.0,
                                  densify_f=1, log=lambda *_: None)
    out = {}
    for w, (P, flag, _) in res.items():
        r = np.hypot(P[..., 0] - CENTRE, P[..., 1] - CENTRE)
        out[w] = (r, flag)
    return out


@unittest.skipUnless(snap_to_surface.HAVE_NUMBA, 'snap_to_surface needs numba')
class SnapToSurfaceTests(unittest.TestCase):
    def test_offset_windings_land_on_their_sheets(self):
        out = run([r + 3.0 for r in SHEETS])
        for w, sheet in zip(sorted(out), SHEETS):
            r, flag = out[w]
            matched = flag == 0
            self.assertGreater(matched.mean(), 0.8)
            self.assertLess(np.abs(r[matched] - sheet).max(), 0.6)

    def test_crowded_windings_never_share_a_sheet(self):
        # Five windings for four sheets, three of them squeezed around 40-50: a nearest-sheet snap puts two of them
        # on the same sheet. Matched windings must keep their order and land on different sheets.
        out = run([44.0, 46.5, 49.0, 61.0, 71.0])
        ws = sorted(out)
        flags = np.stack([out[w][1] for w in ws])
        radii = np.stack([out[w][0] for w in ws])
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                both = (flags[i] == 0) & (flags[j] == 0)
                if both.any():
                    self.assertGreater(np.min(radii[j][both] - radii[i][both]), 5.0)
        matched_r = radii[flags == 0]
        self.assertGreater(matched_r.size, 0)
        self.assertLess(np.min(np.abs(matched_r[:, None] - np.array(SHEETS)[None]), 1).max(), 0.6)


@unittest.skipUnless(snap_to_surface.HAVE_NUMBA, 'snap_to_surface needs numba')
class FeatherTests(unittest.TestCase):
    def test_feather_keeps_matched_nodes_and_never_stacks_fill(self):
        S = sheet_volume().view(ChunkedArray)
        radii = [44.0, 46.5, 49.0, 61.0, 71.0]
        windings = {w: (f'w{w:03d}', cylinder(r), {}) for w, r in enumerate(radii, start=1)}
        kw = dict(rays=256, rmax=100.0, densify_f=1, log=lambda *_: None)
        plain, _ = snap_to_surface.snap(windings, S, lambda z: (CENTRE, CENTRE), (2, 30), **kw)
        res, _ = snap_to_surface.snap(windings, S, lambda z: (CENTRE, CENTRE), (2, 30), feather=1.0, **kw)
        snap_to_surface.drop_stacked_fill(res, workers=1)
        for w in res:
            P, flag, _ = res[w]
            m = plain[w][1] == 0
            self.assertTrue(np.array_equal(flag == 0, m))
            np.testing.assert_array_equal(P[m], plain[w][0][m])          # matched nodes are untouched
            self.assertTrue(np.isfinite(P).all(-1)[m].all())
            self.assertTrue(np.isin(flag, (0, 3, 4)).all())              # matched, out of band or filled; no flag 1
        ws = sorted(res)
        for i, w in enumerate(ws):  # no node of a winding within 1 voxel radially of a neighbour's node at the same grid cell
            for j in (i - 1, i + 1):
                if 0 <= j < len(ws):
                    a, b = res[w][0], res[ws[j]][0]
                    both = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
                    ra = np.hypot(a[..., 0] - CENTRE, a[..., 1] - CENTRE)
                    rb = np.hypot(b[..., 0] - CENTRE, b[..., 1] - CENTRE)
                    self.assertGreater(np.abs(ra - rb)[both].min(), 1.0)


@unittest.skipUnless(snap_to_surface.HAVE_NUMBA, 'snap_to_surface needs numba')
class SnapToSurfaceCliTests(unittest.TestCase):
    def test_cli_writes_tifxyz_meshes_render_ink_can_read(self):
        import tempfile
        import zarr
        from click.testing import CliRunner
        from tifxyz import load_tifxyz, save_tifxyz

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            surface = zarr.open_array(tmp / 'surface.zarr', mode='w', shape=(32, 160, 160), chunks=(8, 160, 160),
                                      dtype='uint8')
            surface[:] = sheet_volume()
            meshes = tmp / 'meshes'
            for w, radius in enumerate(SHEETS, start=1):
                zyxs = cylinder(radius + 3.0)[..., ::-1].copy()
                save_tifxyz(zyxs, str(meshes), f'w{w:03d}_spliced_test', 1.0, 9.6, 'test fit')
            umb = tmp / 'umbilicus.json'
            umb.write_text(json.dumps({'control_points': [{'x': CENTRE, 'y': CENTRE, 'z': 0},
                                                          {'x': CENTRE, 'y': CENTRE, 'z': 32}]}))
            out = tmp / 'snapped'
            result = CliRunner().invoke(snap_to_surface.main, [
                str(meshes), str(out), '--surface', str(tmp / 'surface.zarr'), '--umbilicus', str(umb),
                '--spliced', '--rays', '256', '--threads', '1'])
            self.assertEqual(result.exit_code, 0, result.output)
            for w, radius in enumerate(SHEETS, start=1):
                patch = load_tifxyz(str(out / f'w{w:03d}_spliced_test'))
                zyx = patch.zyxs[patch.valid_vertex_mask].numpy()
                self.assertGreater(len(zyx), 0)
                r = np.hypot(zyx[:, 2] - CENTRE, zyx[:, 1] - CENTRE)
                self.assertLess(np.abs(r - radius).max(), 0.6)
                self.assertEqual(patch.scale.tolist(), [2.0, 2.0])
            report = json.loads((out / 'snap_report.json').read_text())
            self.assertEqual(len(report['windings']), 4)


if __name__ == '__main__':
    unittest.main()
