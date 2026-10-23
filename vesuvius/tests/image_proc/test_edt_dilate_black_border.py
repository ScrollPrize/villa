"""edt-dilate's default black_border=True must not dilate from the volume faces.

The task runs the EDT on the inverted mask, so edt's black_border=True turned every
face of the EDT input into foreground: each non-empty chunk got a `distance`-wide
shell along its faces. The scipy fallback never did this, and ignored black_border=False.
"""
import numpy as np
import pytest

scipy_ndimage = pytest.importorskip("scipy.ndimage")

from vesuvius.image_proc.run.zarr_tasks.tasks import edt_dilate as m

from .test_edt_dilate_halo import _run_chunked


def _ball(shape, centre, dist):
    zz, yy, xx = np.indices(shape)
    c = np.array(centre).reshape(3, 1, 1, 1)
    return ((np.stack([zz, yy, xx]) - c) ** 2).sum(0) <= dist ** 2


BACKENDS = [False] + ([True] if m.HAS_EDT else [])


@pytest.mark.parametrize("use_edt", BACKENDS)
@pytest.mark.parametrize("ch", [(32, 32, 32), (16, 16, 16)])
def test_default_black_border_is_a_plain_dilation(tmp_path, monkeypatch, ch, use_edt):
    monkeypatch.setattr(m, "HAS_EDT", use_edt)
    monkeypatch.setattr(m, "scipy_edt", scipy_ndimage.distance_transform_edt, raising=False)
    vol = np.zeros((32, 32, 32), np.uint8)
    vol[16, 16, 16] = 1
    got = _run_chunked(tmp_path, vol, ch, 3.0, True)
    expected = _ball(vol.shape, (16, 16, 16), 3.0)
    assert expected.sum() == 123
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got, scipy_ndimage.distance_transform_edt(vol == 0) <= 3.0)


@pytest.mark.parametrize("use_edt", BACKENDS)
def test_black_border_false_dilates_from_volume_faces_in_every_chunk(tmp_path, monkeypatch, use_edt):
    # With black_border=False the volume faces (not chunk faces) count as foreground, also
    # in chunks that hold no foreground, so the result does not depend on the chunking.
    monkeypatch.setattr(m, "HAS_EDT", use_edt)
    monkeypatch.setattr(m, "scipy_edt", scipy_ndimage.distance_transform_edt, raising=False)
    vol = np.zeros((12, 12, 12), np.uint8)
    vol[2, 2, 2] = 1
    whole = _run_chunked(tmp_path / "a", vol, (12, 12, 12), 1.0, False)
    chunked = _run_chunked(tmp_path / "b", vol, (4, 4, 4), 1.0, False)
    np.testing.assert_array_equal(chunked, whole)
    assert whole[11, 11, 11] and not whole[6, 6, 6]
