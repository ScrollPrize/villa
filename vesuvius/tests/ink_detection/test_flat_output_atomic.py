"""A failed replacement must not destroy a completed inference image."""
from pathlib import Path

import numpy as np
import pytest
import tifffile

from vesuvius.ink_detection.inference.infer import write_output_tiff


class InterruptedStore:
    shape = (32, 32)

    def __getitem__(self, slices):
        y, x = slices
        if x.start == 16:
            raise OSError("source chunk unavailable")
        return np.ones((y.stop - y.start, x.stop - x.start), dtype=np.float32)


@pytest.mark.parametrize("existing", [False, True])
def test_source_failure_preserves_destination_and_cleans_stage(tmp_path, existing):
    output = tmp_path / "prediction.tif"
    original = None
    if existing:
        tifffile.imwrite(output, np.full((32, 32), 42, dtype=np.uint8))
        original = output.read_bytes()
    with pytest.raises(OSError, match="source chunk unavailable"):
        write_output_tiff(InterruptedStore(), np.ones((32, 32), dtype=np.float32), output, (16, 16))
    if existing:
        assert output.read_bytes() == original
    else:
        assert not output.exists()
    assert set(tmp_path.iterdir()) == ({output} if existing else set())


def test_publish_failure_preserves_destination_and_cleans_stage(tmp_path, monkeypatch):
    output = tmp_path / "prediction.tif"
    tifffile.imwrite(output, np.full((32, 32), 42, dtype=np.uint8))
    original = output.read_bytes()

    def refuse_replace(self, destination):
        assert Path(destination) == output
        raise PermissionError("destination locked")

    monkeypatch.setattr(Path, "replace", refuse_replace)
    with pytest.raises(PermissionError, match="destination locked"):
        write_output_tiff(np.ones((32, 32), dtype=np.float32),
                          np.ones((32, 32), dtype=np.float32), output, (16, 16))
    assert output.read_bytes() == original
    assert set(tmp_path.iterdir()) == {output}


def test_complete_replacement_keeps_tiff_encoding_and_leaves_no_stage(tmp_path):
    output = tmp_path / "prediction.tif"
    tifffile.imwrite(output, np.full((32, 32), 42, dtype=np.uint8))
    write_output_tiff(np.full((32, 32), .5, dtype=np.float32),
                      np.ones((32, 32), dtype=np.float32), output, (16, 16))
    np.testing.assert_array_equal(tifffile.imread(output), np.full((32, 32), 127, dtype=np.uint8))
    with tifffile.TiffFile(output) as result:
        assert result.is_bigtiff
        assert result.pages[0].is_tiled
        assert result.pages[0].compression.name == "LZW"
    assert set(tmp_path.iterdir()) == {output}
