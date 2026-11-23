"""Shared cache operations, including cancellation through Python callbacks."""
import numpy as np
import pytest

from vc_reader import _reader
from vc_reader.reader import ZarrReader, ZarrWriter
from test_zarr_writer_s3 import s3  # shared signed loopback S3 fixture


def options_for(s3, root):
    access = _reader.s3_access("s3://bucket/files", s3.auth.access_key_id,
                              s3.auth.secret_access_key, "", s3.auth.region, s3.auth.endpoint_url)
    return _reader.OpenOptions(access, [str(root)], True, None, False, False,
                               None, 1, 4, "mmap", False)


def test_download_cancel_resume_and_callback_error(s3, tmp_path):
    s3.objects.update({f"files/{i}.bin": bytes([i]) * 17 for i in range(8)})
    options = options_for(s3, tmp_path)
    cancel = _reader.CacheCancellation()
    observed = []

    def progress(state):
        observed.append((state.status, state.completed_files))
        if state.completed_files:
            cancel.cancel()

    first = _reader.download_prefix("s3://bucket/files", options, workers=1,
                                    progress=progress, cancel=cancel)
    assert first.cancelled and first.error == "cancelled"
    assert first.progress.completed_files == 1
    assert observed[0] == ("listing", 0)
    cancel.reset()
    second = _reader.download_prefix("s3://bucket/files", options, workers=4, cancel=cancel)
    assert not second.cancelled and not second.error
    assert second.progress.completed_files == second.progress.total_files == 8
    assert second.progress.completed_bytes == second.progress.total_bytes == 8 * 17
    for i in range(8):
        assert (tmp_path / "s3/bucket/files" / f"{i}.bin").read_bytes() == bytes([i]) * 17
    # Every object is fetched exactly once across the two runs, plus two listings.
    assert s3.methods.count("GET") == 10

    def broken(state):
        if state.completed_files:
            raise ValueError("callback failure")

    with pytest.raises(ValueError, match="callback failure"):
        _reader.download_prefix("s3://bucket/files", options, workers=4, progress=broken)


def test_prefill_remote_then_offline(s3, tmp_path):
    data = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    writer = ZarrWriter.create("s3://bucket/volume", shape=data.shape, chunks=(1, 1, 1),
                               dtype="u1", auth=s3.auth)
    writer[:] = data
    reader = ZarrReader.open("s3://bucket/volume", auth=s3.auth, cache_dir=tmp_path,
                              threads=1, memory_cache=None)
    cancel = _reader.CacheCancellation()

    def progress(done, total):
        assert total == 64
        if done >= 32:
            cancel.cancel()

    first = _reader.prefill(reader._array, cancel, progress)
    assert first.status == _reader.PrefillStatus.Cancelled
    assert 32 <= first.resolved_chunks < 64
    cancel.reset()
    second = _reader.prefill(reader._array, cancel)
    assert second.status == _reader.PrefillStatus.Completed
    assert second.resolved_chunks == 64 and second.error_chunks == 0
    assert first.fetched_chunks + second.fetched_chunks == 64
    before = len(s3.methods)
    third = _reader.prefill(reader._array)
    assert third.fetched_chunks == 0
    offline = ZarrReader.open("s3://bucket/volume", auth=s3.auth, cache_dir=tmp_path, offline=True)
    np.testing.assert_array_equal(offline[:], data)
    assert len(s3.methods) == before
