"""Config contract; optional local snapshot audit without GPU or volume I/O."""
import glob
import json
from pathlib import Path

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.loader import load_config
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_fiber_file


CONFIGS = Path(__file__).resolve().parents[2] / "src/vesuvius/neural_tracing/fiber_trace_3d/configs"
CONFIG = CONFIGS / "train_collection_20260909_5scroll_16x8.json"


def test_collection_training_config():
    raw = json.loads(CONFIG.read_text())
    baseline = json.loads((CONFIGS / "train_s1a_nml_8gpu_fast.json").read_text())
    assert not load_config(CONFIG).embedding.enabled
    assert raw["training"]["learning_rate"] == 0.01
    assert raw["training"]["max_steps"] == 2000000
    assert raw["training"]["optimizer_global_batch_size"] == 128
    assert raw["batch_size"] == 16
    assert raw.pop("sampling_strategy") == "scroll_balanced_arclength"
    for split in ("datasets", "test_datasets"):
        assert all(d["dense_supervision"] is True for d in raw[split])
        assert len({d["scroll_id"] for d in raw[split]}) == 5
    for key in ("test_interval", "checkpoint_interval"):
        assert raw["training"][key] == 1000
        baseline["training"][key] = 1000
    for field in ("datasets", "test_datasets", "embedding"):
        raw.pop(field, None)
        baseline.pop(field, None)
    # The S1a config is also used for LR/continuation experiments; the new
    # collection retains its explicitly asserted initial LR and duration.
    for field in ("run_name", "learning_rate", "max_steps"):
        raw["training"].pop(field)
        baseline["training"].pop(field)
    assert raw == baseline


def test_local_collection_membership_and_parser():
    raw = json.loads(CONFIG.read_text())
    if not Path(raw["datasets"][0]["fiber_glob"]).parent.exists():
        pytest.skip("local collection snapshot is not installed")
    expected = {"datasets": [197, 74, 77, 5, 463, 130],
                "test_datasets": [10, 3, 5, 2, 24, 6]}
    selected = {}
    for split, counts in expected.items():
        selected[split] = set()
        assert len(raw[split]) == 6
        assert len({d["base_volume_path"] for d in raw[split]}) == 5
        for dataset, count in zip(raw[split], counts, strict=True):
            assert dataset["base_volume_scale"] == 1
            assert not any("transform" in key for key in dataset)
            paths = sorted(map(Path, glob.glob(dataset["fiber_glob"])))
            assert len(paths) == count
            for path in paths:
                assert path.suffix == ".json"
                assert path.resolve() not in selected[split]
                selected[split].add(path.resolve())
                source = json.loads(path.read_text()).get("vc_open_data_source_path")
                if source:
                    assert source.replace(
                        "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/",
                        "s3://vesuvius-challenge-open-data/",
                    ) == dataset["base_volume_path"]
                fibers = load_fiber_file(path)
                assert len(fibers) == 1
                assert len(fibers[0].line_points_xyz) >= 2
    assert selected["datasets"].isdisjoint(selected["test_datasets"])
