#!/usr/bin/env python3

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_render_valgrind_ci.py"
SPEC = importlib.util.spec_from_file_location("run_render_valgrind_ci", SCRIPT)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class RenderValgrindCiTest(unittest.TestCase):
    def setUp(self):
        self.identity = {
            "compiler_id": "GNU",
            "compiler_version": "14.2",
            "build_type": "Release",
            "architecture_target": "x86-64-v3",
            "fixture": "serial",
            "scenario": "full_res",
            "width": 96,
            "height": 96,
            "tile_size": 32,
            "repetitions": 1,
            "measured_pixels": 9216,
            "worker_override": 1,
            "valgrind_version": "valgrind-3.22.0",
            "cache_geometry": DRIVER.CACHE_GEOMETRY,
            "benchmark_metadata_schema": 1,
        }
        self.result = {
            "case": "serial/full_res",
            "model_sha256": "model-hash",
            "identity": self.identity,
            "checksum": 123,
            "modeled_runtime_score_ns": 100.0,
        }
        self.reference = {
            "schema_version": 1,
            "model_sha256": "model-hash",
            "tolerance": 0.10,
            "cases": {
                "serial/full_res": {
                    "identity": self.identity,
                    "checksum": 123,
                    "modeled_runtime_score_ns": 100.0,
                }
            },
        }

    def test_callgrind_command_uses_separate_profiles_and_fixed_cache(self):
        command = DRIVER.callgrind_command(
            Path("bench"),
            "parallel",
            "fallback_3",
            Path("metadata.json"),
            Path("callgrind.out"),
            1,
            separate_threads=True,
        )
        self.assertIn("--separate-threads=yes", command)
        self.assertIn("--fair-sched=yes", command)
        self.assertIn(f"--D1={DRIVER.CACHE_GEOMETRY['D1']}", command)
        self.assertEqual(command[-1], "--callgrind")

    def test_drd_command_collects_vector_clocks_and_scheduler_events(self):
        command = DRIVER.drd_command(
            Path("bench"),
            "mixed_correlated",
            Path("metadata.json"),
            Path("drd.log"),
            1,
            10000,
        )
        self.assertIn("--tool=drd", command)
        self.assertIn("--trace-segment=yes", command)
        self.assertIn("--trace-sched=yes", command)
        self.assertIn("--scheduling-quantum=10000", command)
        self.assertNotIn("--callgrind", command)

    def test_reference_gate_accepts_both_ten_percent_bounds(self):
        for score in (90.0, 110.0):
            result = copy.deepcopy(self.result)
            result["modeled_runtime_score_ns"] = score
            DRIVER.check_reference(result, self.reference, 0.10)

    def test_reference_tolerance_is_authoritative_by_default(self):
        self.reference["tolerance"] = 0.05
        self.result["modeled_runtime_score_ns"] = 94.9
        with self.assertRaisesRegex(RuntimeError, "required"):
            DRIVER.check_reference(self.result, self.reference)
        self.result["modeled_runtime_score_ns"] = 95.0
        DRIVER.check_reference(self.result, self.reference)

    def test_explicit_diagnostic_tolerance_can_override_reference(self):
        self.reference["tolerance"] = 0.05
        self.result["modeled_runtime_score_ns"] = 94.0
        DRIVER.check_reference(self.result, self.reference, 0.10)

    def test_invalid_tolerances_are_rejected(self):
        for tolerance in (-0.01, 1.0, float("inf"), float("nan")):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(RuntimeError, "tolerance"):
                    DRIVER.validate_tolerance(tolerance)

    def test_reference_gate_rejects_below_lower_bound(self):
        self.result["modeled_runtime_score_ns"] = 89.9
        with self.assertRaisesRegex(RuntimeError, "required"):
            DRIVER.check_reference(self.result, self.reference, 0.10)

    def test_reference_gate_rejects_above_upper_bound(self):
        self.result["modeled_runtime_score_ns"] = 110.1
        with self.assertRaisesRegex(RuntimeError, "required"):
            DRIVER.check_reference(self.result, self.reference, 0.10)

    def test_reference_gate_rejects_model_and_environment_changes(self):
        changed_model = copy.deepcopy(self.result)
        changed_model["model_sha256"] = "other"
        with self.assertRaisesRegex(RuntimeError, "model hash"):
            DRIVER.check_reference(changed_model, self.reference, 0.10)
        changed_environment = copy.deepcopy(self.result)
        changed_environment["identity"]["compiler_version"] = "15.0"
        with self.assertRaisesRegex(RuntimeError, "environment"):
            DRIVER.check_reference(changed_environment, self.reference, 0.10)

    def test_reference_gate_accepts_valgrind_change_and_records_versions(self):
        result = copy.deepcopy(self.result)
        reference = copy.deepcopy(self.reference)
        result["identity"]["valgrind_version"] = "valgrind-3.26.0"
        DRIVER.check_reference(result, reference, 0.10)
        self.assertEqual(result["reference_valgrind_version"], "valgrind-3.22.0")
        self.assertEqual(result["observed_valgrind_version"], "valgrind-3.26.0")
        self.assertTrue(result["valgrind_version_changed"])

    def test_reference_gate_keeps_valgrind_diagnostics_on_score_failure(self):
        result = copy.deepcopy(self.result)
        reference = copy.deepcopy(self.reference)
        result["identity"]["valgrind_version"] = "valgrind-3.26.0"
        result["modeled_runtime_score_ns"] = 110.1
        with self.assertRaisesRegex(RuntimeError, "required"):
            DRIVER.check_reference(result, reference, 0.10)
        self.assertEqual(result["reference_valgrind_version"], "valgrind-3.22.0")
        self.assertEqual(result["observed_valgrind_version"], "valgrind-3.26.0")
        self.assertTrue(result["valgrind_version_changed"])

    def test_reference_gate_requires_valgrind_metadata(self):
        for source in ("reference", "observed"):
            with self.subTest(source=source):
                result = copy.deepcopy(self.result)
                reference = copy.deepcopy(self.reference)
                identity = (
                    reference["cases"]["serial/full_res"]["identity"]
                    if source == "reference"
                    else result["identity"]
                )
                identity["valgrind_version"] = ""
                with self.assertRaisesRegex(RuntimeError, "Valgrind version"):
                    DRIVER.check_reference(result, reference, 0.10)

    def test_pair_requires_matching_valgrind_versions(self):
        callgrind = {
            "case": "parallel/full_res",
            "metadata": {},
            "valgrind_version": "valgrind-3.25.1",
        }
        drd = {
            "case": "parallel/full_res",
            "metadata": {},
            "valgrind_version": "valgrind-3.26.0",
            "trace": {
                "unmatched_futex_waits": 0,
                "unresolved_happens_before": 0,
            },
        }
        with (
            mock.patch.object(DRIVER, "_verify_manifest_files"),
            self.assertRaisesRegex(RuntimeError, "different Valgrind versions"),
        ):
            DRIVER._validate_pair(callgrind, drd)

    def test_reference_gate_checks_exact_checksum(self):
        self.result["checksum"] = 124
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            DRIVER.check_reference(self.result, self.reference, 0.10)

    def test_trace_completeness_requires_all_dependencies(self):
        complete = SimpleNamespace(
            unmatched_waits=0, unresolved_happens_before=0, events=[object()]
        )
        self.assertTrue(DRIVER._trace_is_complete(complete))
        complete.unmatched_waits = 1
        self.assertFalse(DRIVER._trace_is_complete(complete))
        complete.unmatched_waits = 0
        complete.unresolved_happens_before = 1
        self.assertFalse(DRIVER._trace_is_complete(complete))

    def test_serial_score_is_normalized_by_repetitions(self):
        callgrind = {
            "profiles": {
                "1": {
                    "Ir": 80,
                    "Dr": 20,
                    "Dw": 10,
                    "D1mr": 2,
                    "D1mw": 1,
                    "DLmr": 1,
                    "DLmw": 0,
                    "Bcm": 2,
                    "Bim": 1,
                }
            },
            "metadata": {"repetitions": 2},
        }
        event_model = {
            "feature_names": [
                "non_data_instructions",
                "data_reads",
                "data_writes",
                "l1_data_misses",
                "last_level_data_misses",
                "branch_misses",
                "branch_weighted_l1_misses",
            ],
            "coefficients_ns": [1.0] * 7,
            "stall_overlap_fraction": 0.0,
        }
        expected_features = 50 + 20 + 10 + 3 + 1 + 3 + (3 * 3 / 80)
        engine = mock.MagicMock()
        engine.__enter__.return_value = engine
        engine.model_profile_costs.return_value = (
            {1: expected_features},
            expected_features,
        )
        with mock.patch.object(DRIVER, "NativeReplayEngine", return_value=engine):
            score, replay = DRIVER.estimate_score(
                callgrind, None, {"event_cost_model": event_model}, Path("unused")
            )
        self.assertAlmostEqual(score, expected_features / 2)
        self.assertIsNone(replay)

    def test_atomic_json_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            DRIVER.write_json_atomic(path, {"complete": True})
            self.assertEqual(json.loads(path.read_text()), {"complete": True})
            self.assertEqual(list(path.parent.glob(".*.tmp-*")), [])

    def test_freeze_model_requires_explicit_unpromoted_approval(self):
        calibration = {
            "renderer_inputs_used": False,
            "candidate_accepted": False,
            "parameters": {"cross_thread_release_ns": 12.5},
            "event_cost_model": {
                "feature_names": list(DRIVER.DATA_READ_FEATURE_NAMES),
                "coefficients_ns": [float(index) for index in range(7)],
                "stall_overlap_fraction": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "calibration.json"
            output = root / "model.json"
            source.write_text(json.dumps(calibration))
            args = SimpleNamespace(
                calibration=source,
                output=output,
                model_id="test-model",
                allow_unpromoted=False,
            )
            with self.assertRaisesRegex(RuntimeError, "not accepted"):
                DRIVER.freeze_model(args)
            args.allow_unpromoted = True
            DRIVER.freeze_model(args)
            model = json.loads(output.read_text())
            self.assertEqual(model["model_id"], "test-model")
            self.assertEqual(model["cross_thread_release_ns"], 12.5)
            self.assertFalse(model["timing_claims_enabled"])

    def test_set_tolerance_preserves_all_reference_cases(self):
        reference = copy.deepcopy(self.reference)
        for fixture in ("serial", "parallel"):
            for scenario in DRIVER.SCENARIOS:
                reference["cases"][f"{fixture}/{scenario}"] = {
                    "identity": {"fixture": fixture, "scenario": scenario},
                    "checksum": 123,
                    "modeled_runtime_score_ns": 100.0,
                }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(reference))
            args = SimpleNamespace(reference=path, output=None, tolerance=0.05)
            DRIVER.set_tolerance(args)
            updated = json.loads(path.read_text())
            self.assertEqual(updated["tolerance"], 0.05)
            reference["tolerance"] = 0.05
            self.assertEqual(updated, reference)


if __name__ == "__main__":
    unittest.main()
