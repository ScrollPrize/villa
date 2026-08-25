"""Tests for Volume.download_inklabel placeholder reporting.

Regression coverage: when the ink label cannot be loaded, Volume substitutes a
blank array for real data. That substitution was only reported when verbose=True
(the default is False), and the reported reason was the fsspec FileNotFoundError,
which carries only the URL - so a TLS or network failure was indistinguishable
from the file genuinely not being published.
"""

import numpy as np
import pytest

from vesuvius.data import volume as volume_module
from vesuvius.data.volume import Volume

URL = "https://example.invalid/scrolls/1/segments/54keV_7.91um/20230827161847.zarr/"
LABEL_URL = "https://example.invalid/scrolls/1/segments/54keV_7.91um/20230827161847_inklabels.png"


def _bare_segment(verbose=False):
    """A Volume with just the attributes download_inklabel touches - no network."""
    v = Volume.__new__(Volume)
    v.type = "segment"
    v.url = URL
    v.verbose = verbose
    v.inklabel = None
    return v


def _fail_with(monkeypatch, exc):
    def boom(*args, **kwargs):
        raise exc
    monkeypatch.setattr(volume_module.fsspec, "open", boom)


def _tls_failure():
    """What fsspec actually raises for a certificate problem: the URL, cause buried."""
    cause = OSError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    err = FileNotFoundError(LABEL_URL)
    err.__cause__ = cause
    return err


class TestInkLabelPlaceholderIsReported:
    def test_warns_even_though_verbose_is_false(self, monkeypatch, capsys):
        """The default path must not silently hand back fabricated data."""
        v = _bare_segment(verbose=False)
        _fail_with(monkeypatch, _tls_failure())
        v.download_inklabel()
        out = capsys.readouterr().out
        assert "could not load ink label" in out.lower()
        assert "does NOT contain real data" in out

    def test_reports_the_underlying_cause_not_just_the_url(self, monkeypatch, capsys):
        """A TLS failure must not read as 'the file is not published'."""
        v = _bare_segment(verbose=False)
        _fail_with(monkeypatch, _tls_failure())
        v.download_inklabel()
        out = capsys.readouterr().out
        assert "CERTIFICATE_VERIFY_FAILED" in out, "the real cause was swallowed"

    def test_placeholder_shape_is_stated(self, monkeypatch, capsys):
        v = _bare_segment(verbose=False)
        _fail_with(monkeypatch, _tls_failure())
        v.download_inklabel()
        out = capsys.readouterr().out
        assert "(1, 1)" in out

    def test_still_returns_an_array_so_callers_do_not_break(self, monkeypatch):
        """Deliberately unchanged: the fallback still yields an array, not None."""
        v = _bare_segment(verbose=False)
        _fail_with(monkeypatch, _tls_failure())
        v.download_inklabel()
        assert isinstance(v.inklabel, np.ndarray)
        assert int(np.sum(v.inklabel)) == 0

    def test_missing_url_still_reported(self, capsys):
        v = _bare_segment()
        v.url = None
        v.download_inklabel()
        assert "URL is not set" in capsys.readouterr().out
