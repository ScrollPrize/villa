"""tolerate_config: bringing a stored checkpoint configuration onto the schema."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint_migrations import tolerate_checkpoint_config, tolerate_config
from config import Config


DEFAULTS = Config().as_dict()


def test_unknown_keys_are_dropped_and_missing_keys_defaulted_with_notes():
    assert tolerate_config(dict(DEFAULTS), defaults=DEFAULTS) == (DEFAULTS, [])
    stored = {**DEFAULTS, "influence_enabled": True, "patch_erode_patches": 7}
    del stored["input_use_tracks"]
    config, notes = tolerate_config(stored, defaults=DEFAULTS)
    assert config == {**DEFAULTS, "patch_erode_patches": 7}
    assert notes == [
        "drops configuration keys the schema no longer has: influence_enabled",
        "predates configuration keys, which take their defaults: "
        f"input_use_tracks={DEFAULTS['input_use_tracks']!r}",
    ]
    # A value the schema cannot interpret has no default to fall back on.
    with pytest.raises(ValueError, match="Invalid value for dense_spacing_mode"):
        tolerate_config({**DEFAULTS, "dense_spacing_mode": "phase"}, defaults=DEFAULTS)


def test_every_configuration_field_of_a_checkpoint_is_normalised():
    stale = {**DEFAULTS, "influence_enabled": False}
    checkpoint = {"cfg": stale, "requested_config": stale, "resolved_config": stale}
    updated, notes = tolerate_checkpoint_config(checkpoint)
    assert "influence_enabled" in checkpoint["cfg"]  # the input is not mutated
    assert all(updated[f] == DEFAULTS for f in ("cfg", "requested_config", "resolved_config"))
    assert notes == ["drops configuration keys the schema no longer has: influence_enabled"]
    assert tolerate_checkpoint_config({"cfg": None}) == ({"cfg": None}, [])
