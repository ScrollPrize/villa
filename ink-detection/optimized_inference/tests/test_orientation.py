"""FORCE_REVERSE=auto picks the order from which end of the window looks like air."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orientation import air_end, end_texture  # noqa: E402


class _Source:
    def __init__(self, arr):
        self._a = arr
        self.shape = arr.shape

    def read_roi(self, y1, y2, x1, x2):
        return self._a[y1:y2, x1:x2]


def _stack(air_first, h=256, w=256, c=62, seed=0):
    rng = np.random.default_rng(seed)
    a = np.empty((h, w, c), np.uint8)
    for z in range(c):
        sd = 2 + 30 * z / (c - 1)  # air (smooth) -> papyrus (textured)
        a[:, :, z] = np.clip(rng.normal(60 + z, sd, (h, w)), 1, 255)
    return a if air_first else a[:, :, ::-1].copy()


def test_air_at_the_low_end():
    assert air_end(_Source(_stack(air_first=True))) == "low"


def test_air_at_the_high_end():
    assert air_end(_Source(_stack(air_first=False))) == "high"


def test_same_texture_at_both_ends_is_undecided():
    rng = np.random.default_rng(1)
    a = np.clip(rng.normal(80, 15, (256, 256, 62)), 1, 255).astype(np.uint8)
    assert air_end(_Source(a)) is None


def test_no_data_is_undecided():
    empty = np.zeros((256, 256, 62), np.uint8)
    assert end_texture(_Source(empty)) is None
    assert air_end(_Source(empty)) is None


def test_too_few_layers_is_undecided():
    assert end_texture(_Source(_stack(True, c=20))) is None


def _env(monkeypatch, value, **extra):
    for k, v in {"MODEL": "m", "START_LAYER": "2", "END_LAYER": "64", "STEP": "inference",
                 "SURFACE_VOLUME_ZARR": "/tmp/sv.zarr", "IMAGE_ROLE": "gpu", **extra}.items():
        monkeypatch.setenv(k, v)
    if value is None:
        monkeypatch.delenv("FORCE_REVERSE", raising=False)
    else:
        monkeypatch.setenv("FORCE_REVERSE", value)
    pytest.importorskip("huggingface_hub")  # entrypoint's own imports
    import entrypoint
    return entrypoint.parse_env()


def test_force_reverse_env_values(monkeypatch):
    assert (lambda i: (i.force_reverse, i.auto_reverse))(_env(monkeypatch, None)) == (False, False)
    assert (lambda i: (i.force_reverse, i.auto_reverse))(_env(monkeypatch, "true")) == (True, False)
    assert (lambda i: (i.force_reverse, i.auto_reverse))(_env(monkeypatch, "AUTO")) == (False, True)
    assert (lambda i: (i.force_reverse, i.auto_reverse))(_env(monkeypatch, "false")) == (False, False)


def test_auto_rejected_for_partitioned_s3(monkeypatch):
    with pytest.raises(ValueError, match="FORCE_REVERSE=auto"):
        _env(monkeypatch, "auto", NUM_PARTS="2", SURFACE_VOLUME_ZARR="s3://b/sv.zarr")
    assert _env(monkeypatch, "auto", NUM_PARTS="2").auto_reverse
