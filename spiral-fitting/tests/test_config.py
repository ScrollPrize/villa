import json
import pytest
from config import Config


def test_mapping_and_json_overrides_and_validation(tmp_path):
    changed = Config({"optimizer_learning_rate": 0.25})
    assert changed.optimizer_learning_rate == 0.25
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"optimizer_learning_rate": 0.5}))
    assert Config(profile).optimizer_learning_rate == 0.5

    with pytest.raises(ValueError, match="Unknown"):
        Config({"not_a_setting": 1})
    with pytest.raises(ValueError, match="Invalid value"):
        Config({"optimizer_learning_rate": "fast"})
    with pytest.raises(ValueError, match="Out-of-range"):
        Config({"optimizer_learning_rate": -1})
    with pytest.raises(ValueError, match="Invalid value"):
        Config({"dense_spacing_mode": "unknown"})
    with pytest.raises(ValueError, match="Invalid vector length"):
        Config({"dense_spacing_pair_m_short": [1]})
    with pytest.raises(ValueError):
        Config({"track_max_tortuosity": "unlimited"})


def test_gap_capacity_and_numerical_floor_are_explicit_and_validated():
    catalog = Config.catalog()
    defaults = catalog["defaults"]
    assert defaults["model_gap_expander_capacity_windings"] > (
        defaults["model_gap_expander_num_windings"])
    with pytest.raises(ValueError, match="capacity_windings"):
        Config({"model_gap_expander_capacity_windings": 2})
    with pytest.raises(ValueError, match="min_gap"):
        Config({"model_gap_expander_min_gap": 0.0})
    with pytest.raises(ValueError, match="min_gap"):
        Config({"model_gap_expander_min_gap": 16.0})
