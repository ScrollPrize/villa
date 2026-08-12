import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "evaluate_render_attribution_sensitivity.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_render_attribution_sensitivity", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

import run_thread_sync_replay as REPLAY


class AttributionSensitivityTest(unittest.TestCase):
    def events(self):
        return [
            REPLAY.TraceEvent(0, 1, "thread_start"),
            REPLAY.TraceEvent(
                1, 1, "hb_segment", dependencies=[(0, "program_order")]
            ),
            REPLAY.TraceEvent(
                2, 1, "work_quantum", dependencies=[(1, "program_order")]
            ),
            REPLAY.TraceEvent(
                3, 1, "hb_segment", dependencies=[(2, "program_order")]
            ),
        ]

    def test_policy_matrix_is_complete(self):
        self.assertEqual(len(MODULE.POLICIES), 9)
        self.assertEqual(
            {policy.placement for policy in MODULE.POLICIES},
            {"front", "equal", "back"},
        )
        self.assertEqual(
            {policy.residual_window_weight for policy in MODULE.POLICIES},
            {0.0, 0.5, 1.0},
        )

    def test_partial_only_zero_weight_preserves_cost(self):
        events = [REPLAY.TraceEvent(0, 1, "hb_segment")]
        REPLAY.assign_costs(events, {1: 125.0}, 0.0, "equal")
        self.assertEqual(events[0].duration, 125.0)

    def test_positive_cost_without_eligible_event_fails(self):
        events = [REPLAY.TraceEvent(0, 1, "thread_start")]
        with self.assertRaisesRegex(RuntimeError, "no eligible event"):
            REPLAY.assign_costs(events, {1: 125.0}, 0.5, "equal")

    def test_every_policy_preserves_exact_thread_cost(self):
        for policy in MODULE.POLICIES:
            events = self.events()
            REPLAY.assign_costs(
                events,
                {1: 125.0},
                policy.residual_window_weight,
                policy.placement,
            )
            MODULE.assert_cost_preservation(events, {1: 125.0})

    def test_policy_evaluation_order_does_not_change_results(self):
        def evaluate(policies):
            events = self.events()
            results = {}
            for policy in policies:
                REPLAY.assign_costs(
                    events,
                    {1: 125.0},
                    policy.residual_window_weight,
                    policy.placement,
                )
                results[policy.policy_id] = REPLAY.simulate(
                    events, 1, "fifo"
                )["modeled_makespan"]
            return results

        self.assertEqual(evaluate(MODULE.POLICIES), evaluate(reversed(MODULE.POLICIES)))

    def test_summary_retains_baseline_and_selects_nothing(self):
        predictions = []
        for scenario in MODULE.SCENARIOS:
            for workers in MODULE.WORKERS:
                predictions.append(
                    {
                        "scenario": scenario,
                        "workers": workers,
                        "measured_ns": 100.0 / workers,
                        "predicted_ns": {
                            policy.policy_id: 100.0 / workers
                            for policy in MODULE.POLICIES
                        },
                    }
                )
        report = MODULE.summarize(predictions)
        self.assertFalse(report["policy_selected"])
        self.assertEqual(report["baseline_policy"], MODULE.BASELINE_POLICY)
        for policy in report["policies"].values():
            self.assertEqual(
                policy["summary"]["runtime_rms_delta_from_baseline_points"],
                0.0,
            )

    def test_hash_verification_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text(json.dumps({"value": 1}))
            provenance = {
                "artifacts": {str(path): MODULE.sha256_file(path)}
            }
            MODULE.verify_frozen_inputs(provenance)
            path.write_text(json.dumps({"value": 2}))
            with self.assertRaisesRegex(RuntimeError, "input changed"):
                MODULE.verify_frozen_inputs(provenance)

    def test_supplied_event_model_must_match_embedded_model(self):
        event_model = {"coefficients_ns": {"Ir": 1.0}}
        sync_model = {"event_cost_model": event_model}
        MODULE.validate_event_model(sync_model, event_model.copy())
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            MODULE.validate_event_model(
                sync_model, {"coefficients_ns": {"Ir": 2.0}}
            )

    def test_model_comparison_splits_single_and_many_worker_maxima(self):
        predictions = []
        for scenario in MODULE.SCENARIOS:
            for workers in MODULE.WORKERS:
                factor = 1.0
                if scenario == "fallback_3" and workers == 1:
                    factor = 1.30
                if scenario == "mixed_shuffled" and workers == 7:
                    factor = 1.40
                predictions.append(
                    {
                        "scenario": scenario,
                        "workers": workers,
                        "measured_ns": 100.0 / workers,
                        "comparison_predicted_ns": {
                            "candidate": factor * 100.0 / workers
                        },
                    }
                )
        report = MODULE.summarize_model_comparisons(
            predictions, ("candidate",)
        )["candidate"]
        self.assertEqual(
            report["maximum_runtime_error_workers_1"]["scenario"],
            "fallback_3",
        )
        self.assertAlmostEqual(
            report["maximum_runtime_error_workers_1"][
                "absolute_error_percent"
            ],
            30.0,
        )
        self.assertEqual(
            report["maximum_runtime_error_workers_2_to_7"]["scenario"],
            "mixed_shuffled",
        )
        self.assertAlmostEqual(
            report["maximum_runtime_error_workers_2_to_7"][
                "absolute_error_percent"
            ],
            40.0,
        )


if __name__ == "__main__":
    unittest.main()
