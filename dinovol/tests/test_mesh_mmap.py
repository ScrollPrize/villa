import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dinovol_2.dataset.object_labels import (
    MeshSegment, load_mesh_segments_cache, mmap_mesh_segments, save_mesh_segments_cache,
)


def segment(seed, h=70, w=90, ink=True):
    rng = np.random.default_rng(seed)
    zyx = rng.random((h, w, 3), dtype=np.float32) * 100
    valid = rng.random((h, w)) > 0.2
    zyx[~valid] = np.nan
    ink_state = np.where(valid, rng.choice([-1, 0, 1], size=(h, w)), 0).astype(np.int8)
    if not ink:
        ink_state[ink_state == 1] = 0
    seg = MeshSegment(segment_id=f"seg{seed}", zyx=zyx, valid=valid, ink_state=ink_state,
                      bbox_lo=np.nanmin(zyx.reshape(-1, 3), 0), bbox_hi=np.nanmax(zyx.reshape(-1, 3), 0),
                      has_ink=bool((ink_state == 1).any()))
    seg.ensure_spatial_index()
    return seg


class MeshMmapTests(unittest.TestCase):
    def test_mmapped_segments_match_the_cache_and_pickle_by_path(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            cache = Path(tmp) / "PHerc_test_scale2.npz"
            save_mesh_segments_cache(cache, [segment(0), segment(1, ink=False)])
            expected = load_mesh_segments_cache(cache)
            mapped = mmap_mesh_segments(cache)
            self.assertTrue((Path(tmp) / "PHerc_test_scale2.npz.mmap" / "segments.json").is_file())
            fields = MeshSegment._MAPPED + ("segment_id", "bbox_lo", "bbox_hi", "has_ink", "n_valid_points",
                                            "n_ink_points", "tile_size")
            for want, got in zip(expected, mapped):
                for name in fields:
                    np.testing.assert_array_equal(getattr(got, name), getattr(want, name), err_msg=name)
                self.assertIsInstance(got.zyx, np.memmap)
                self.assertFalse(got.zyx.flags.writeable)
                blob = pickle.dumps(got)
                self.assertLess(len(blob), 4096)  # paths and scalars only
                back = pickle.loads(blob)
                self.assertIsInstance(back.zyx, np.memmap)
                for name in fields:
                    np.testing.assert_array_equal(getattr(back, name), getattr(want, name), err_msg=name)
            # A second load reuses the sidecar.
            again = mmap_mesh_segments(cache)
            self.assertEqual(again[0].mmap_dir, mapped[0].mmap_dir)

    def test_in_memory_segments_still_pickle_their_arrays(self):
        seg = segment(2)
        back = pickle.loads(pickle.dumps(seg))
        np.testing.assert_array_equal(back.zyx, seg.zyx)
        self.assertIsNone(back.mmap_dir)


if __name__ == "__main__":
    unittest.main()
