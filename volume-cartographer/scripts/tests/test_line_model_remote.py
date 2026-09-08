import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_line_model_remote import (
    binary_artifacts, inventory, merge_payloads, result_signature, validate_manifest, verified_run)
from benchmark_line_model_prefetch import parse_metric_output, write_json


class RemoteBenchmarkTests(unittest.TestCase):
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
