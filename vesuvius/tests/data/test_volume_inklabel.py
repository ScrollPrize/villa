"""Tests for Volume.download_inklabel placeholder reporting.

Regression coverage: when the ink label cannot be loaded, Volume substitutes a
blank array for real data. That substitution was only reported when verbose=True
(the default is False), and the reported reason was the fsspec FileNotFoundError,
which carries only the URL - so a TLS or network failure was indistinguishable
from the file genuinely not being published.
"""

import numpy as np

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


class _ClientResponseErrorLike(Exception):
    """Mirrors the attribute aiohttp.ClientResponseError exposes: .status.

    A stand-in rather than the real class, because aiohttp's constructor requires a
    request_info and history that carry no meaning here. The only attribute the code
    under test reads is .status.
    """

    def __init__(self, status, message):
        super().__init__(f"{status}, message={message!r}")
        self.status = status
        self.message = message


def _absent_404():
    """What fsspec ACTUALLY raises for a label that is not published.

    Verified live against dl.ash2txt.org on 2026-08-31: a real 404 surfaces as
    FileNotFoundError chained from aiohttp's ClientResponseError with status 404.
    An earlier version of this test used a bare FileNotFoundError with no cause,
    which no real 404 ever produces, so it could not catch a wrong split.
    """
    cause = _ClientResponseErrorLike(404, "Not Found")
    err = FileNotFoundError(LABEL_URL)
    err.__cause__ = cause
    return err


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
        assert "could not fetch the ink label" in out.lower()
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


class TestMissingIsDistinguishedFromFailed:
    """Most segments publish no ink label at all. That is normal and must not be
    reported in the same words as a request that failed."""

    def test_absent_label_is_reported_as_a_note(self, monkeypatch, capsys):
        v = _bare_segment(verbose=False)
        _fail_with(monkeypatch, _absent_404())  # a real 404: cause present, status 404
        v.download_inklabel()
        out = capsys.readouterr().out
        assert "no ink label is published" in out
        assert "not real data" in out
        assert "Warning" not in out, "an absent label is not a failure"

    def test_a_decode_failure_is_not_reported_as_absent(self, monkeypatch, capsys):
        """A published-but-unreadable label is not the same as no label.

        Only a missing file is "not published". If the download succeeds and Image.open
        raises, there is no chained cause, so a naive `cause is None` check would announce
        the label as absent when it is in fact there.
        """
        _fail_with(monkeypatch, ValueError("cannot identify image file"))
        v = _bare_segment(verbose=False)
        v.download_inklabel()
        out = capsys.readouterr().out
        assert "no ink label is published" not in out
        assert "Warning" in out

    def test_failed_request_is_reported_as_a_warning(self, monkeypatch, capsys):
        v = _bare_segment(verbose=False)
        _fail_with(monkeypatch, _tls_failure())  # chained cause = the request failed
        v.download_inklabel()
        out = capsys.readouterr().out
        assert "Warning" in out
        assert "the label may exist" in out
        assert "CERTIFICATE_VERIFY_FAILED" in out
