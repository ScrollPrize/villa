"""zarr_tasks pyramids must carry the per-level scale that OME-NGFF 0.4 requires.

write_multiscales_metadata (used by the threshold, transpose, scale and
recompress tasks) wrote `datasets` entries with only a `path`. The 0.4 spec says
each dataset MUST have `coordinateTransformations` with exactly one `scale`, so
the pyramids failed validation and readers saw no level scales. build_pyramid
halves every axis per level, so level i must be scaled by 2**i on each axis,
which is what the merge task already writes.
"""
import numpy as np
import zarr
from numcodecs import Blosc

from vesuvius.image_proc.run.zarr_tasks.utils import (
    build_pyramid,
    create_level_dataset,
    write_multiscales_metadata,
)


def test_each_level_has_a_scale_matching_the_pyramid(tmp_path):
    root = str(tmp_path / "pyr.zarr")
    lvl0 = create_level_dataset(root, "0", (32, 32, 32), (8, 8, 8), np.uint8, Blosc(cname="zstd", clevel=1))
    lvl0[:] = np.arange(32 ** 3, dtype=np.uint32).reshape(32, 32, 32).astype(np.uint8)
    build_pyramid(root, ["z", "y", "x"], num_levels=3, num_workers=1)
    write_multiscales_metadata(root, ["z", "y", "x"], num_levels=3)

    g = zarr.open_group(root, mode="r")
    ms = g.attrs["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == ["z", "y", "x"]
    for i, ds in enumerate(ms["datasets"]):
        transforms = ds["coordinateTransformations"]
        assert [t["type"] for t in transforms] == ["scale"]
        scale = transforms[0]["scale"]
        assert scale == [float(2 ** i)] * 3
        # the scale must describe the array actually stored at that level
        assert tuple(s * f for s, f in zip(g[ds["path"]].shape, scale)) == g["0"].shape
