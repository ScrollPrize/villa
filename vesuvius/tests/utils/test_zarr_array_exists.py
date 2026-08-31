"""Tests for zarr_array_exists.

Regression coverage: the S3 branch hardcoded anon=False, so a machine with no AWS
credentials could not see arrays in the open-data bucket, which is published
public-read. Every exception was then swallowed into False, making "I could not
look" indistinguishable from "it is not there" - and wait_for_zarr_creation()
turns that into a five-minute wait followed by "the array was never created".
"""

import os

import pytest

from vesuvius.utils.io import zarr_utils
from vesuvius.utils.io.zarr_utils import zarr_array_exists


class _FS:
    def __init__(self, result=None, raises=None):
        self._result, self._raises = result, raises

    def exists(self, path):
        if self._raises is not None:
            raise self._raises
        return self._result


def _filesystems(monkeypatch, by_anon):
    """Stub fsspec.filesystem so each anon= value gets its own behaviour."""
    def factory(protocol, anon=False, **kw):
        assert protocol == "s3"
        return by_anon[anon]
    monkeypatch.setattr(zarr_utils.fsspec, "filesystem", factory)


class TestLocalPaths:
    def test_present(self, tmp_path):
        (tmp_path / ".zarray").write_text("{}")
        assert zarr_array_exists(str(tmp_path)) is True

    def test_absent(self, tmp_path):
        assert zarr_array_exists(str(tmp_path)) is False


    def test_accepts_a_path_object(self, tmp_path):
        """os.PathLike must work. main caught the AttributeError and returned False
        for every Path, so this asserts we are not worse than main for that input."""
        arr = tmp_path / "present.zarr"
        arr.mkdir()
        (arr / ".zarray").write_text("{}")
        assert zarr_array_exists(arr) is True
        assert zarr_array_exists(tmp_path / "absent.zarr") is False


class TestS3Paths:
    def test_falls_back_to_anonymous_when_credentials_are_missing(self, monkeypatch):
        """The open-data bucket is public-read; no credentials must still work."""
        _filesystems(monkeypatch, {
            False: _FS(raises=RuntimeError("Unable to locate credentials")),
            True: _FS(result=True),
        })
        assert zarr_array_exists("s3://vesuvius-challenge-open-data/x.zarr") is True

    def test_absent_object_is_still_false(self, monkeypatch):
        _filesystems(monkeypatch, {False: _FS(result=False), True: _FS(result=False)})
        assert zarr_array_exists("s3://vesuvius-challenge-open-data/nope.zarr") is False

    def test_raises_when_existence_cannot_be_determined(self, monkeypatch):
        """'I could not look' must not be reported as 'it is not there'."""
        boom = OSError("network is unreachable")
        _filesystems(monkeypatch, {False: _FS(raises=boom), True: _FS(raises=boom)})
        with pytest.raises(RuntimeError) as excinfo:
            zarr_array_exists("s3://vesuvius-challenge-open-data/x.zarr")
        assert "Could not determine" in str(excinfo.value)
        assert "network is unreachable" in str(excinfo.value)

    def test_credentialed_access_is_tried_first(self, monkeypatch):
        """A user with real credentials must not be silently downgraded to anonymous."""
        seen = []

        def factory(protocol, anon=False, **kw):
            seen.append(anon)
            return _FS(result=True)

        monkeypatch.setattr(zarr_utils.fsspec, "filesystem", factory)
        assert zarr_array_exists("s3://private-bucket/x.zarr") is True
        assert seen == [False], "anonymous should only be a fallback"
