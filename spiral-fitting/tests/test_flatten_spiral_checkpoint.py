import json
from pathlib import Path

import pytest

from flatten_spiral_checkpoint import (
    MODEL_CONFIG_KEYS,
    _checkpoint_config,
    _resolve_lasagna,
    _resolve_umbilicus,
    _store_surface,
)


def test_legacy_checkpoint_model_config_gets_current_aliases():
    legacy = {key: index for index, key in enumerate(MODEL_CONFIG_KEYS)}
    config = _checkpoint_config({"cfg": legacy})
    for key in MODEL_CONFIG_KEYS:
        assert config[f"model_{key}"] == legacy[key]


def test_current_checkpoint_model_config_is_preserved():
    current = {
        f"model_{key}": index for index, key in enumerate(MODEL_CONFIG_KEYS)
    }
    config = _checkpoint_config({"cfg": current})
    assert config == current


def test_resolve_umbilicus_prefers_explicit_path(tmp_path):
    path = tmp_path / "umbilicus.json"
    path.write_text(json.dumps({"control_points": []}))
    checkpoint = tmp_path / "checkpoint.ckpt"
    assert _resolve_umbilicus(checkpoint, path) == path.resolve()


def test_resolve_umbilicus_finds_checkpoint_ancestor(tmp_path):
    path = tmp_path / "umbilicus.json"
    path.write_text(json.dumps({"control_points": []}))
    checkpoint = tmp_path / "spiral_output" / "run" / "checkpoint.ckpt"
    assert _resolve_umbilicus(checkpoint, None) == path.resolve()


def test_resolve_lasagna_requires_service_and_config(tmp_path):
    (tmp_path / "fit_service.py").write_text("")
    config = tmp_path / "configs" / "flatten_fast_nofilter.json"
    config.parent.mkdir()
    config.write_text("{}")
    assert _resolve_lasagna(tmp_path) == (
        (tmp_path / "fit_service.py").resolve(),
        config.resolve(),
    )


def test_checkpoint_config_reports_missing_fields():
    with pytest.raises(ValueError, match="missing model configuration"):
        _checkpoint_config({"cfg": {}})


def _surface(root: Path) -> Path:
    """A minimal tifxyz segment: a couple of files in a directory."""
    surface = root / "segment-0001"
    (surface / "nested").mkdir(parents=True)
    (surface / "meta.json").write_text('{"width": 4}', encoding="utf-8")
    (surface / "nested" / "x.tif").write_bytes(bytes([0, 1, 2]))
    return surface


def test_store_surface_falls_back_to_copying_where_symlinks_are_refused(
        tmp_path, monkeypatch):
    """Symlinks need a privilege on Windows and do not exist on exFAT.

    The object store is private to the run and is only ever read, so a copy
    has to serve where the symlink is refused.
    """
    attempted = []

    def refuse(self, target, target_is_directory=False):
        attempted.append(Path(self))
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(Path, "symlink_to", refuse)

    surface = _surface(tmp_path)
    store = tmp_path / "objects"
    ref = _store_surface(surface, store)

    segment = store / ref["type"] / ref["hash"].removeprefix("md5:") / "segment-0001" / "segment"
    assert attempted, "the symlink has to be tried before copying"
    assert segment.is_dir() and not segment.is_symlink()
    assert (segment / "meta.json").read_text(encoding="utf-8") == '{"width": 4}'
    assert (segment / "nested" / "x.tif").read_bytes() == bytes([0, 1, 2])
    assert json.loads(
        (segment.parent / "object.json").read_text(encoding="utf-8")) == ref


def test_store_surface_hash_does_not_depend_on_how_the_segment_is_stored(
        tmp_path, monkeypatch):
    """The manifest hash is over the segment's bytes, not over the store."""
    linked = _store_surface(_surface(tmp_path / "a"), tmp_path / "a" / "objects")

    def refuse(self, target, target_is_directory=False):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(Path, "symlink_to", refuse)
    copied = _store_surface(_surface(tmp_path / "b"), tmp_path / "b" / "objects")

    assert linked == copied

