from pathlib import Path

import numpy as np

import fit_spiral
import tracks
from runners import run_sweep


def _sweep_args(tmp_path: Path, *extra: str):
    return run_sweep.build_parser().parse_args([
        "--dataset", str(tmp_path / "dataset"),
        "--ink-volume", str(tmp_path / "ink"),
        "--output", str(tmp_path / "output"),
        "--config-folder", str(tmp_path / "configs"),
        "--sweep-config", str(tmp_path / "sweep.json"),
        *extra,
    ])


def test_sweep_child_defaults_to_six_cpu_threads(tmp_path):
    args = _sweep_args(tmp_path)

    command = run_sweep.child_command(
        args, "baseline", tmp_path / "baseline.json", (0,))

    assert command[command.index("--num-threads") + 1] == "6"


def test_sweep_child_accepts_cpu_thread_override(tmp_path):
    args = _sweep_args(tmp_path, "--num-threads", "3")

    command = run_sweep.child_command(
        args, "baseline", tmp_path / "baseline.json", (0,))

    assert command[command.index("--num-threads") + 1] == "3"


def test_trusted_geometry_query_respects_configured_cpu_threads(monkeypatch):
    class RecordingTree:
        workers = None

        def query(self, points, *, k, distance_upper_bound, workers):
            self.workers = workers
            return np.zeros(len(points)), np.zeros(len(points), dtype=np.intp)

    tree = RecordingTree()
    monkeypatch.setattr(fit_spiral.torch, "get_num_threads", lambda: 6)

    result = fit_spiral._query_near_trusted_geometry(
        np.zeros((2, 3)), tree, threshold=4.0)

    assert tree.workers == 6
    np.testing.assert_array_equal(result, np.ones(2, dtype=bool))


def test_track_exclusion_query_respects_configured_cpu_threads(monkeypatch):
    class RecordingTree:
        workers = None

        def query(self, points, *, k, distance_upper_bound, workers):
            self.workers = workers
            return np.full(len(points), np.inf), np.zeros(
                len(points), dtype=np.intp)

    tree = RecordingTree()
    monkeypatch.setattr(tracks.torch, "get_num_threads", lambda: 6)

    result = tracks._track_points_far_from_anchors_mask(
        np.zeros((2, 3)), tree, threshold=4.0)

    assert tree.workers == 6
    np.testing.assert_array_equal(result, np.ones(2, dtype=bool))
