"""fetch_band: the band's rows per frame, and a sparse copy that holds exactly the chunks with those rows."""
import json

import h5py
import numpy as np
import pytest

from vc_tomography import _core
from vc_tomography import fetch_band
from test_htj2k_reader import DATA, _chunk, _write
from test_projection_download import _FakeSFTP


def test_band_rows_follow_the_stage_and_the_row_sign():
    # pixel 1 um: 1 mm of stage travel moves the band 1000 rows
    assert fetch_band.band_rows(-256.0, -256.0, 1.0, 2368, 16, 0, -1) == (1175, 1193)
    assert fetch_band.band_rows(-256.0, -256.0, 1.0, 2368, 16, 32, -1) == (1143, 1225)
    # a higher stage puts the sample on lower rows with row sign -1, on higher rows with +1
    lo, hi = fetch_band.band_rows(-255.5, -256.0, 1.0, 2368, 16, 0, -1)
    assert (lo, hi) == (675, 693)
    assert fetch_band.band_rows(-255.5, -256.0, 1.0, 2368, 16, 0, +1) == (1675, 1693)
    # clipped at the detector's edges, empty off it
    assert fetch_band.band_rows(-254.82, -256.0, 1.0, 2368, 16, 32, -1) == (0, 45)
    lo, hi = fetch_band.band_rows(-250.0, -256.0, 1.0, 2368, 16, 32, -1)
    assert lo >= hi
    sel = fetch_band.chunk_selection([-256.0, -250.0, -255.5], 7, -256.0, 1.0, 2368, 16, 0, -1)
    assert sel == {7: (1175, 1193), 9: (675, 693)}


@pytest.mark.skipif(not _core.htj2k_float_frames_supported(), reason="VC_TOMOGRAPHY_OPENJPH=OFF")
def test_sparse_copy_holds_the_band_chunks_alone(tmp_path, monkeypatch):
    rows, cols, chunk_rows = 768, 256, 256
    source = tmp_path / "raw" / "scan" / "det.h5"
    source.parent.mkdir(parents=True)
    _write(source, _chunk((DATA / "unsigned16_97.j2k").read_bytes(), chunk_rows, cols, sample_bytes=2),
           rows=rows, cols=cols, frames=3, chunk_rows=chunk_rows, dtype="uint16")
    sftp = _FakeSFTP()
    monkeypatch.setattr(fetch_band, "_connect", lambda spec: sftp)
    # frame 0 inside the first chunk, frame 1 across the first two, frame 2 off the detector
    job = dict(tile="T", role="projection", entry="5.1", file="scan/det.h5", dataset="data", rows=rows,
               frames={"0": [10, 20], "1": [250, 270]}, id="job")
    out = tmp_path / "out"
    result = fetch_band._run_job(job, str(out), "esrf:" + str(tmp_path / "raw"), str(tmp_path / "raw"), 0)
    copy = out / "scan" / "det.h5"
    record = json.loads((out / "scan" / "det.h5.band.json").read_text())
    assert not record["whole"] and len(record["fetched"]["offsets"]) == 3
    assert record["source"] == "esrf:" + str(tmp_path / "raw") + "/scan/det.h5"
    assert copy.stat().st_size == source.stat().st_size
    with h5py.File(source, "r") as f:
        chunk = f["data"].id.get_chunk_info(0).size
    assert 3 * chunk <= result["bytes"] < 4 * chunk  # three chunks and a block of metadata, of nine chunks

    full = _core.RawReader([str(source)], "data", rows, cols, decoder="openjph")
    sparse = _core.RawReader([str(copy)], "data", rows, cols, decoder="openjph")
    zero = np.zeros(1, np.int32)
    for frame, (lo, hi) in ((0, (10, 20)), (1, (250, 270)), (1, (0, 512))):
        args = (zero, np.array([frame], np.int32), np.array([lo], np.int32), hi - lo)
        np.testing.assert_array_equal(sparse.read_rows(*args), full.read_rows(*args))
    for frame, row in ((0, 300), (1, 600), (2, 0)):  # chunks that were not fetched fail to decode
        with pytest.raises(Exception, match="blosc2"):
            sparse.read_rows(zero, np.array([frame], np.int32), np.array([row], np.int32), 4)

    assert fetch_band._run_job(job, str(out), "esrf:x", str(tmp_path / "raw"), 0)["resumed"]
    assert not fetch_band._run_job(dict(job, id="other"), str(out), "esrf:" + str(tmp_path / "raw"),
                                   str(tmp_path / "raw"), 0)["resumed"]

    several = dict(job, id="two bands", row_windows={"0": [[10, 20], [540, 545]]})
    del several["frames"]
    fetch_band._run_job(several, str(out), "esrf:" + str(tmp_path / "raw"), str(tmp_path / "raw"), 0)
    sparse = _core.RawReader([str(copy)], "data", rows, cols, decoder="openjph")
    for row in (10, 540):
        args = (zero, zero, np.array([row], np.int32), 1)
        np.testing.assert_array_equal(sparse.read_rows(*args), full.read_rows(*args))
    with pytest.raises(Exception, match="blosc2"):
        sparse.read_rows(zero, zero, np.array([300], np.int32), 1)
