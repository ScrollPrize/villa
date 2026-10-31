"""--task recompress --inplace must keep the source array on a trailing slash or a failed copy."""
import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import RecompressConfig, RecompressTask

# On zarr 3, --inplace cannot create its temp array yet (#2069).
pytestmark = pytest.mark.skipif(zarr.__version__.startswith("3."), reason="in-place recompress needs zarr 2")
DATA = (np.arange(12**3) % 251 + 1).astype(np.uint8).reshape(12, 12, 12)


def _recompress_inplace(tmp_path, path_suffix="", truncate=None):
    path = tmp_path / "vol.zarr"
    source = zarr.open(str(path), mode="w", shape=DATA.shape, chunks=(4, 4, 4), dtype=np.uint8,
                       dimension_separator="/")
    source[:] = DATA
    if truncate:
        (path / truncate).write_bytes((path / truncate).read_bytes()[:10])
    config = RecompressConfig(input_zarr=str(path) + path_suffix, output_zarr=None, num_workers=1, inplace=True)
    RecompressTask(config).run()


def test_trailing_slash_recompresses_the_array(tmp_path):
    _recompress_inplace(tmp_path, path_suffix="/")
    out = zarr.open(str(tmp_path / "vol.zarr"), mode="r")
    assert out.compressor.cname == "zstd"
    np.testing.assert_array_equal(out[:], DATA)


def test_failed_copy_puts_the_copied_chunks_back(tmp_path):
    # One worker copies the chunks in order, so the 26 before the truncated last one are moved first.
    with pytest.raises(RuntimeError, match="Recompression failed"):
        _recompress_inplace(tmp_path, truncate="2/2/2")
    np.testing.assert_array_equal(zarr.open(str(tmp_path / "vol.zarr"), mode="r")[:8], DATA[:8])
