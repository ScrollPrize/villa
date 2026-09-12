import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_line_model_remote import (
    binary_artifacts, inventory, merge_payloads, payload_difference, result_signature,
    summarize, validate_manifest, verified_run)
from benchmark_line_model_prefetch import parse_metric_output, write_json


class RemoteBenchmarkTests(unittest.TestCase):
    def test_summary_includes_optional_projection_and_observed_traffic_metrics(self):
        metrics = dict.fromkeys(('trace_wall_s', 'remote_bytes', 'span_trace_ms',
                                'reinit_ms', 'tail_trace_ms', 'report_prefetch_ms'), '1')
        rows = [dict(mode=mode, process_wall_s=1, metrics=metrics,
                     profile=dict(model_prefetch_replans=2),
                     observed_payload_difference=dict(extra_observed_bytes=30))
                for mode in ('cold', 'warm')]
        summary = summarize(rows)
        self.assertEqual(summary['cold']['model_prefetch_replans']['mean'], 2)
        self.assertEqual(summary['warm']['extra_observed_bytes']['median'], 30)
        self.assertNotIn('model_prefetch_reference_plans', summary['cold'])

    def test_observed_traffic_comparison_excludes_metadata(self):
        reference = {'normal/array/0/0/0': {'bytes': 100},
                     'normal/array/0/0/1': {'bytes': 20}}
        current = {'normal/array/0/0/0': {'bytes': 100},
                   'normal/array/0/0/2': {'bytes': 30},
                   'normal/model.json': {'bytes': 40},
                   'normal/.lasagna-zarr-metadata/array/.zarray': {'bytes': 50}}
        self.assertEqual(payload_difference(current, reference), dict(
            extra_observed_objects=1, extra_observed_bytes=30,
            absent_reference_objects=1, absent_reference_bytes=20))

    def test_library_only_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = ['bin/metric', 'core/libvc_lasagna.so', 'core/libvc_fiber_tracer.so',
                        'core/libvc_core.so', 'utils/libutils.so',
                        'utils/libutils_c3d_codec.so', 'libs/c3d/libc3d.so']
            for name in relative:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'original')
            initial = binary_artifacts(root / 'bin/metric')
            (root / 'core/libvc_lasagna.so').write_bytes(b'changed')
            updated = binary_artifacts(root / 'bin/metric')
            self.assertEqual(initial['bin/metric'], updated['bin/metric'])
            self.assertNotEqual(initial, updated)

    def test_metric_protocol_includes_dirty_span_and_drain(self):
        metrics, profile, warmup = parse_metric_output(
            'native_trace2cp_gui_stages dirty_span=6 reinit_ms=12.5\n'
            'native_trace2cp_remote remote_bytes=42 pending_requests=0\n'
            'native_trace2cp_profile candidates=123 generations=10\n')
        self.assertEqual(metrics['dirty_span'], '6')
        self.assertEqual(metrics['pending_requests'], '0')
        self.assertEqual(profile['candidates'], 123)

    def test_remote_marker_and_source_containment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'model.json'
            write_json(root / 'lasagna-remote.json', dict(
                artifact_url='https://example.invalid/model', manifest_file=path.name))
            for route in ('../escape', '/tmp/escape', 'https://example.invalid/other'):
                write_json(path, dict(groups=dict(nx=dict(zarr=route))))
                with self.assertRaises(ValueError):
                    validate_manifest(path)
            write_json(path, dict(groups=dict(nx=dict(zarr='nx.zarr/4'))))
            self.assertEqual(validate_manifest(path)[1], 'https://example.invalid/model/model.json')

    def test_cache_inventory_checks_all_bytes_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'chunk').write_bytes(b'first')
            known = inventory(root)
            merge_payloads(known, inventory(root))
            (root / 'chunk').write_bytes(b'other')
            with self.assertRaises(ValueError):
                merge_payloads(known, inventory(root))
            (root / 'link').symlink_to(root / 'chunk')
            with self.assertRaises(ValueError):
                inventory(root)

    def test_signature_excludes_only_known_times(self):
        trace = dict(format='vc_gui_fiber_reoptimization_exact_v1',
                     optimization=dict(message='ceres_solve_ms=1.2 reason=ok', points=[1, 2, 3]))
        profile = dict(candidates=123, generations=10)
        initial = result_signature(trace, profile)
        trace['optimization']['message'] = 'ceres_solve_ms=9.9 reason=ok'
        self.assertEqual(initial, result_signature(trace, profile))
        trace['optimization']['points'][0] = 2
        self.assertNotEqual(initial, result_signature(trace, profile))

    def test_incomplete_comparison_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'input').mkdir()
            write_json(root / 'configuration.json', dict(input_hashes={}, trials=1))
            write_json(root / 'results.json', [])
            with self.assertRaises(ValueError):
                verified_run(root)


if __name__ == '__main__':
    unittest.main()
