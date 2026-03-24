from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from ink.core import (
    BatchMeta,
    DataBundle,
    EvalReport,
    Experiment,
    RunFS,
    build_run_dir,
    build_run_id,
    slugify_name,
    to_plain,
)


def _load_yamlish(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return json.loads(text)
    return yaml.safe_load(text)


@dataclass(frozen=True)
class _Recipe:
    kind: str
    value: int


class CoreTypesTests(unittest.TestCase):
    def test_experiment_and_core_types_are_dataclass_friendly(self):
        experiment = Experiment(
            name="erm_baseline",
            data=_Recipe(kind="data", value=1),
            model=_Recipe(kind="model", value=2),
            loss=_Recipe(kind="loss", value=3),
            objective=_Recipe(kind="objective", value=4),
            runtime=_Recipe(kind="runtime", value=5),
            augment=_Recipe(kind="augment", value=6),
        )
        bundle = DataBundle(
            train_loader="train",
            eval_loader="val",
            in_channels=62,
            group_counts=[3, 4],
        )
        meta = BatchMeta(segment_ids=["seg-1"])

        self.assertEqual(asdict(experiment)["name"], "erm_baseline")
        self.assertEqual(bundle.group_counts, [3, 4])
        self.assertIsNone(meta.valid_mask)

    def test_to_plain_handles_nested_dataclasses(self):
        experiment = Experiment(
            name="erm_baseline",
            data=_Recipe(kind="data", value=1),
            model=_Recipe(kind="model", value=2),
            loss=_Recipe(kind="loss", value=3),
            objective=_Recipe(kind="objective", value=4),
            runtime=_Recipe(kind="runtime", value=5),
            augment=_Recipe(kind="augment", value=6),
        )

        plain = to_plain({"experiment": experiment, "path": Path("runs/demo")})

        self.assertEqual(plain["experiment"]["model"]["kind"], "model")
        self.assertEqual(plain["path"], "runs/demo")


class RunFSTests(unittest.TestCase):
    def test_initialization_creates_run_layout_and_snapshot(self):
        experiment = Experiment(
            name="erm_baseline",
            data=_Recipe(kind="data", value=1),
            model=_Recipe(kind="model", value=2),
            loss=_Recipe(kind="loss", value=3),
            objective=_Recipe(kind="objective", value=4),
            runtime=_Recipe(kind="runtime", value=5),
            augment=_Recipe(kind="augment", value=6),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(Path(tmpdir) / "runs" / "demo_run", experiment)

            self.assertTrue(run_fs.eval_dir.is_dir())
            self.assertTrue(run_fs.ckpt_dir.is_dir())
            self.assertTrue(run_fs.artifacts_dir.is_dir())

            snapshot = _load_yamlish(run_fs.experiment_path)
            self.assertEqual(snapshot["name"], "erm_baseline")
            self.assertEqual(snapshot["runtime"]["kind"], "runtime")

    def test_logging_writes_history_summary_and_latest_eval(self):
        experiment = Experiment(
            name="erm_baseline",
            data=_Recipe(kind="data", value=1),
            model=_Recipe(kind="model", value=2),
            loss=_Recipe(kind="loss", value=3),
            objective=_Recipe(kind="objective", value=4),
            runtime=_Recipe(kind="runtime", value=5),
            augment=_Recipe(kind="augment", value=6),
        )
        report = EvalReport(summary={"val/dice": 0.71}, by_segment={"seg-1": {"val/dice": 0.71}})

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(Path(tmpdir) / "runs" / "demo_run", experiment)
            run_fs.log_train_epoch(0, {"train/loss": 0.2})
            run_fs.log_eval_epoch(0, report)

            history_lines = run_fs.history_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(history_lines), 2)
            self.assertEqual(json.loads(history_lines[0])["split"], "train")
            self.assertEqual(json.loads(history_lines[0])["components"]["train/loss"], 0.2)
            self.assertEqual(json.loads(history_lines[1])["summary"]["val/dice"], 0.71)

            summary = _load_yamlish(run_fs.summary_path)
            latest = _load_yamlish(run_fs.eval_dir / "latest.yaml")
            self.assertEqual(summary["split"], "val")
            self.assertEqual(latest["val/dice"], 0.71)

    def test_best_and_last_checkpoints_are_persisted(self):
        experiment = Experiment(
            name="erm_baseline",
            data=_Recipe(kind="data", value=1),
            model=_Recipe(kind="model", value=2),
            loss=_Recipe(kind="loss", value=3),
            objective=_Recipe(kind="objective", value=4),
            runtime=_Recipe(kind="runtime", value=5),
            augment=_Recipe(kind="augment", value=6),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(Path(tmpdir) / "runs" / "demo_run", experiment)
            low_report = EvalReport(summary={"val/dice": 0.40})
            high_report = EvalReport(summary={"val/dice": 0.80})

            run_fs.save_last(model_state={"w": [1]}, optimizer_state={"lr": 1e-4}, epoch=1)
            first_best = run_fs.maybe_save_best(
                "val/dice",
                low_report,
                model_state={"w": [1]},
                optimizer_state={"lr": 1e-4},
                epoch=1,
            )
            second_best = run_fs.maybe_save_best(
                "val/dice",
                high_report,
                model_state={"w": [2]},
                optimizer_state={"lr": 5e-5},
                epoch=2,
            )

            self.assertTrue((run_fs.ckpt_dir / "last.pt").is_file())
            self.assertEqual(first_best, run_fs.ckpt_dir / "best.pt")
            self.assertEqual(second_best, run_fs.ckpt_dir / "best.pt")
            self.assertEqual(run_fs.best_metric, 0.80)

            best_summary = _load_yamlish(run_fs.eval_dir / "best.yaml")
            self.assertEqual(best_summary["val/dice"], 0.80)


class RunLayoutTests(unittest.TestCase):
    def test_slugify_name_normalizes_and_falls_back(self):
        self.assertEqual(slugify_name(" erm baseline "), "erm_baseline")
        self.assertEqual(slugify_name("erm/baseline:v1"), "erm_baseline_v1")
        self.assertEqual(slugify_name("..."), "run")

    def test_build_run_id_uses_fixed_date_and_suffix(self):
        run_id = build_run_id(
            "erm baseline",
            now=datetime(2026, 3, 16, 12, 30, 0),
            suffix="ab12cd",
        )

        self.assertEqual(run_id, "2026-03-16_erm_baseline_ab12cd")

    def test_build_run_dir_joins_root_with_run_id(self):
        run_dir = build_run_dir(
            Path("runs"),
            "groupdro baseline",
            now=datetime(2026, 3, 16, 12, 30, 0),
            suffix="ef34gh",
        )

        self.assertEqual(run_dir, Path("runs/2026-03-16_groupdro_baseline_ef34gh"))


if __name__ == "__main__":
    unittest.main()
