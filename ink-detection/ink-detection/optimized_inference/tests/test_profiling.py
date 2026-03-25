import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import profiling  # noqa: E402


class ProfilingTests(unittest.TestCase):
    def test_parse_detailed_partition_selector(self):
        self.assertEqual(profiling.parse_detailed_partition_selector("first")["mode"], "first")
        self.assertTrue(profiling.should_profile_partition("first", 0))
        self.assertFalse(profiling.should_profile_partition("first", 1))
        self.assertTrue(profiling.should_profile_partition("all", 7))
        self.assertTrue(profiling.should_profile_partition("1,3,5", 3))
        self.assertFalse(profiling.should_profile_partition("1,3,5", 2))

    def test_schema_and_flags_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = profiling.WorkflowProfiler(
                level="off",
                sample_interval_ms=1000,
                raw_root=temp_dir,
                local_root=temp_dir,
                step_name="prepare",
                template_name="prepare-surface-volume",
                part_id=None,
                metadata={"pod_name": "pod-a", "workflow_name": "wf"},
                runtime_parameters={"tile_size": 64},
            )
            profiler.add_duration("download_seconds", 1.5, flag=profiling.FLAG_APPROXIMATE)
            profiler.increment_counter("s3_download_bytes", 1234)
            profiler.flush("succeeded")

            summary = json.loads(Path(temp_dir, profiling.PROFILE_SUMMARY_NAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], profiling.SCHEMA_VERSION)
            self.assertEqual(summary["metrics"]["download_seconds"]["flag"], profiling.FLAG_APPROXIMATE)
            self.assertEqual(summary["metrics"]["s3_download_bytes"]["value"], 1234)
            self.assertEqual(summary["metrics"]["total_wall_seconds"]["flag"], profiling.FLAG_EXACT)

    def test_sampler_fallback_marks_metrics_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(profiling, "psutil", None), mock.patch.object(profiling, "pynvml", None):
                profiler = profiling.WorkflowProfiler(
                    level="basic",
                    sample_interval_ms=1000,
                    raw_root=temp_dir,
                    local_root=temp_dir,
                    step_name="inference",
                    template_name="run-ink-detection-partition",
                    part_id=0,
                    metadata={"pod_name": "pod-b", "workflow_name": "wf"},
                    runtime_parameters={},
                )
                profiler.flush("succeeded")

            summary = json.loads(Path(temp_dir, profiling.PROFILE_SUMMARY_NAME).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["metrics"]["process_cpu_utilization_percent_avg"]["flag"],
                profiling.FLAG_UNAVAILABLE,
            )
            self.assertEqual(
                summary["metrics"]["gpu_utilization_percent_avg"]["flag"],
                profiling.FLAG_UNAVAILABLE,
            )

    def test_aggregate_dedupes_retries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir, "raw")
            out_dir = Path(temp_dir, "out")
            first = raw_root / "inference" / "part-000" / "pod-retry-a"
            second = raw_root / "inference" / "part-000" / "pod-retry-b"
            first.mkdir(parents=True, exist_ok=True)
            second.mkdir(parents=True, exist_ok=True)

            failed_payload = {
                "schema_version": profiling.SCHEMA_VERSION,
                "status": "failed",
                "identifiers": {"template_name": "run-ink-detection-partition", "part_id": 0, "pod_name": "pod-retry-a"},
                "timestamps": {"end_utc": "2026-03-08T09:00:00Z"},
                "metrics": {
                    "total_wall_seconds": profiling.metric_entry(40.0, profiling.FLAG_EXACT, profiling.SEMANTICS_DURATION),
                    "partition_tiles": profiling.metric_entry(100.0, profiling.FLAG_EXACT, profiling.SEMANTICS_COUNTER),
                },
            }
            success_payload = {
                "schema_version": profiling.SCHEMA_VERSION,
                "status": "succeeded",
                "identifiers": {"template_name": "run-ink-detection-partition", "part_id": 0, "pod_name": "pod-retry-b"},
                "timestamps": {"end_utc": "2026-03-08T10:00:00Z"},
                "metrics": {
                    "total_wall_seconds": profiling.metric_entry(25.0, profiling.FLAG_EXACT, profiling.SEMANTICS_DURATION),
                    "partition_tiles": profiling.metric_entry(100.0, profiling.FLAG_EXACT, profiling.SEMANTICS_COUNTER),
                    "forward_seconds": profiling.metric_entry(12.0, profiling.FLAG_EXACT, profiling.SEMANTICS_DURATION),
                },
            }
            Path(first, profiling.PROFILE_SUMMARY_NAME).write_text(json.dumps(failed_payload), encoding="utf-8")
            Path(second, profiling.PROFILE_SUMMARY_NAME).write_text(json.dumps(success_payload), encoding="utf-8")

            profiling.aggregate_workflow_profiling(raw_root, out_dir)
            summary = json.loads(Path(out_dir, profiling.WORKFLOW_SUMMARY_JSON).read_text(encoding="utf-8"))

            self.assertEqual(summary["selected_records"], 1)
            self.assertEqual(summary["partition_count"], 1)
            self.assertEqual(summary["top_slowest_partitions"][0]["pod_name"], "pod-retry-b")


if __name__ == "__main__":
    unittest.main()
