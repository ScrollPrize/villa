"""Correctness comparison must ignore timings, never geometry or decisions."""
import argparse
import copy
import io
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_line_model_prefetch import analyze, canonical_trace, digest, normalized_message, validate_cross_variants
from benchmark_line_model_prefetch import (
    Handler, configured_readers_match, drain_trial_requests, file_sha256,
    positive_finite_seconds, require_drained_requests, write_json)


class ExactComparisonTest(unittest.TestCase):
    def test_only_human_diagnostic_is_excluded(self):
        original = {'format':'vc_gui_fiber_reoptimization_exact_v1',
                    'points':['8000000000000000'],
                    'optimization':{'message':'timing 1.0', 'converged':True},
                    'control_points':[{'accepted_native':True, 'failure_detail':''}]}
        other = copy.deepcopy(original)
        other['optimization']['message'] = 'timing 2.0'
        self.assertEqual(digest(canonical_trace(original)), digest(canonical_trace(other)))
        self.assertIn('message', original['optimization'])
        other['points'][0] = '0000000000000000'
        self.assertNotEqual(digest(canonical_trace(original)), digest(canonical_trace(other)))
        other = copy.deepcopy(original)
        other['control_points'][0]['accepted_native'] = False
        self.assertNotEqual(digest(canonical_trace(original)), digest(canonical_trace(other)))
        other = copy.deepcopy(original)
        other['optimization']['converged'] = False
        self.assertNotEqual(digest(canonical_trace(original)), digest(canonical_trace(other)))

    def test_diagnostic_mask_preserves_solver_status_and_counts(self):
        message = ('reinit-reopt+global      1.874e+01      8    1.577e+00    1.433e+00\n'
                   'calls=9 ceres_solve_ms=3.0 chunk_prefetch_ms=0.0 materialize_ms=2.0 '
                   'requested_chunks=2538 chunks_read=2538 total_ms=2.0\n'
                   'Time (in seconds):\n'
                   'Preprocessor                         0.000717\n'
                   '  Residual only evaluation           0.000071 (8)\n'
                   'Total                                0.013093\n'
                   'Termination: CONVERGENCE\n')
        timing_changed = message.replace('1.874e+01', '2.874e+01').replace(
            'ceres_solve_ms=3.0', 'ceres_solve_ms=9.0').replace('0.000071', '0.900071')
        self.assertEqual(normalized_message(message), normalized_message(timing_changed))
        for old, new in [('CONVERGENCE', 'FAILURE'), ('(8)', '(9)'),
                         ('1.433e+00', '1.533e+00'), ('chunks_read=2538', 'chunks_read=2539')]:
            self.assertNotEqual(normalized_message(message), normalized_message(message.replace(old, new)))

    def test_native_trace_has_no_ignored_fields(self):
        trace = {'format':'native_trace', 'report':{'message':'failed'}}
        self.assertEqual(canonical_trace(trace), trace)

    def test_cross_variant_metrics_checked_without_trace_files(self):
        baseline = {'identical_result_fields':{'segments':28, 'native_segments':28},
                    'identical_candidate_count':100, 'identical_generation_count':10,
                    'exact_trace_sha256':None, 'normalized_message_sha256':None,
                    'provenance':{'source_sha256':'source', 'manifest_sha256':{'fiber':'manifest'}}}
        validate_cross_variants({'disabled':baseline, 'enabled':copy.deepcopy(baseline)})
        for field in ('identical_candidate_count', 'identical_generation_count'):
            changed = copy.deepcopy(baseline)
            changed[field] += 1
            with self.assertRaises(ValueError):
                validate_cross_variants({'disabled':baseline, 'enabled':changed})
        changed = copy.deepcopy(baseline)
        changed['identical_result_fields']['native_segments'] -= 1
        with self.assertRaises(ValueError):
            validate_cross_variants({'disabled':baseline, 'enabled':changed})

    def test_requested_reader_configuration_must_be_confirmed(self):
        args = ['--model-readers', '064']
        self.assertFalse(configured_readers_match(args, {}))
        self.assertFalse(configured_readers_match(args, {'configured_max':'64', 'adaptive':'1'}))
        self.assertFalse(configured_readers_match(args, {'configured_max':'32', 'adaptive':'0'}))
        self.assertTrue(configured_readers_match(args, {'configured_max':'64', 'adaptive':'0'}))
        self.assertTrue(configured_readers_match([], {}))


class RecordedRunTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root/'run'
        self.directory.mkdir()
        self.source = self.root/'input.json'
        write_json(self.source, {'line_points':[[1, 2, 3]], 'control_points':[]})
        self.manifests = {}
        for kind in ('fiber', 'normal'):
            folder = self.root/kind
            folder.mkdir()
            manifest = folder/f'{kind}.lasagna.json'
            write_json(manifest, {'kind':kind})
            self.manifests[kind] = manifest
            cache = self.directory/'cache-1'/kind
            cache.mkdir(parents=True)
            (cache/manifest.name).write_bytes(manifest.read_bytes())
            write_json(cache/'lasagna-remote.json', {'artifact_url':'http://example.invalid'})
        source_array = self.manifests['fiber'].parent/'channel.zarr'
        source_array.mkdir()
        (source_array/'0.0.0').write_bytes(bytes(range(16)))
        (source_array/'.zarray').write_bytes(b'{"shape":[2,2,4]}')
        self.payload = self.directory/'cache-1/fiber/channel.zarr/0.0.0'
        self.payload.parent.mkdir()
        self.payload.write_bytes((source_array/'0.0.0').read_bytes())
        self.metadata = self.directory/'cache-1/fiber/.lasagna-zarr-metadata/channel.zarr/.zarray'
        self.metadata.parent.mkdir(parents=True)
        self.metadata.write_bytes((source_array/'.zarray').read_bytes())
        self.workload = {
            'source':str(self.source), 'source_sha256':file_sha256(self.source),
            'manifests':{kind:str(path) for kind, path in self.manifests.items()},
            'manifest_sha256':{kind:file_sha256(path) for kind, path in self.manifests.items()}}
        self.configuration = {
            'binary':str(self.root/'historical-binary'), 'binary_sha256':'recorded-binary-hash',
            'model_prefetch':'1', 'extra_args':['--gui-reoptimize'],
            'threads':4, 'delay_ms':50.0, 'bandwidth_mib':1.0, 'drain_timeout_s':5.0,
            'modes':['cold'], 'trials':1, 'save_traces':False}
        self.row = {
            'mode':'cold', 'trial':1, 'returncode':0, 'requests_drained':True,
            'metrics':{key:'1' for key in (
                'segments', 'native_segments', 'lasagna_fallback_segments',
                'cspline_fallback_segments', 'native_tails', 'lasagna_tails', 'points')},
            'profile':{key:1.0 for key in (
                'candidates', 'generations', 'prediction_batch_s', 'prediction_materialize_s',
                'normal_batch_s', 'normal_materialize_s', 'candidate_score_s', 'frontier_s',
                'prune_s', 'start_sample_s')}}
        self.persist_recording()
        write_json(self.directory/'cold-1.requests.json', [])

    def persist_recording(self):
        write_json(self.directory/'workload.json', self.workload)
        write_json(self.directory/'configuration.json', self.configuration)
        write_json(self.directory/'results.json', [self.row])

    def test_complete_cache_walk_and_provenance_are_reported_without_logged_gets(self):
        summary = analyze(self.directory, {})
        self.assertEqual(summary['byte_verified_get_objects'], 0)
        self.assertEqual(summary['byte_verified_cache_objects'], 2)
        self.assertEqual(summary['byte_verified_seeded_manifests'], 2)
        self.assertEqual(summary['byte_verified_cache_bytes'],
                         self.payload.stat().st_size + self.metadata.stat().st_size)
        provenance = summary['provenance']
        self.assertEqual(provenance['source_sha256'], file_sha256(self.source))
        self.assertEqual(provenance['verified_source_path'], str(self.source.resolve()))
        self.assertEqual(provenance['manifest_sha256'], self.workload['manifest_sha256'])
        self.assertEqual(provenance['verified_manifest_paths'], self.workload['manifests'])
        self.assertEqual(provenance['binary_sha256'], 'recorded-binary-hash')
        self.assertEqual(provenance['model_prefetch'], '1')
        self.assertEqual(provenance['extra_args'], ['--gui-reoptimize'])
        for field in ('threads', 'delay_ms', 'bandwidth_mib', 'drain_timeout_s'):
            self.assertEqual(provenance[field], self.configuration[field])

    def test_current_fiber_provenance_is_rehashed(self):
        self.source.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'fiber source provenance changed'):
            analyze(self.directory, {})

    def test_each_current_manifest_provenance_is_rehashed(self):
        for kind, path in self.manifests.items():
            with self.subTest(kind=kind):
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                with self.assertRaisesRegex(ValueError, f'{kind} manifest provenance changed'):
                    analyze(self.directory, {})
                path.write_bytes(original)

    def test_manifest_override_must_match_recorded_bytes(self):
        replacement = self.root/'replacement.json'
        replacement.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'fiber manifest provenance changed'):
            analyze(self.directory, {'fiber':replacement})

    def test_seeded_manifest_bytes_must_match_original_recorded_hash(self):
        for kind, manifest in self.manifests.items():
            with self.subTest(kind=kind):
                seeded = self.directory/'cache-1'/kind/manifest.name
                original = seeded.read_bytes()
                seeded.write_bytes(original + b' ')
                with self.assertRaisesRegex(ValueError, 'seeded cached manifest provenance changed'):
                    analyze(self.directory, {})
                seeded.write_bytes(original)

    def test_seeded_manifest_must_exist(self):
        seeded = self.directory/'cache-1/normal'/self.manifests['normal'].name
        seeded.unlink()
        with self.assertRaisesRegex(ValueError, 'seeded cached manifest is missing'):
            analyze(self.directory, {})

    def test_identical_manifest_override_reports_verified_path_and_original_seed(self):
        original = self.manifests['fiber']
        replacement = original.with_name('renamed.lasagna.json')
        replacement.write_bytes(original.read_bytes())
        summary = analyze(self.directory, {'fiber':replacement})
        self.assertEqual(summary['provenance']['verified_manifest_paths']['fiber'], str(replacement))
        self.assertEqual(summary['provenance']['recorded_manifest_paths']['fiber'], str(original))
        self.assertEqual(summary['byte_verified_cache_objects'], 2)

    def test_cross_variant_workload_identity_is_required_but_binary_may_change(self):
        baseline = analyze(self.directory, {})
        changed = copy.deepcopy(baseline)
        changed['provenance']['binary_sha256'] = 'intentional-new-binary'
        changed['provenance']['model_prefetch'] = '0'
        validate_cross_variants({'before':baseline, 'after':changed})
        for field in ('source_sha256', 'manifest_sha256'):
            with self.subTest(field=field):
                changed = copy.deepcopy(baseline)
                changed['provenance'][field] = 'changed-workload'
                with self.assertRaisesRegex(ValueError, 'provenance differs across variants'):
                    validate_cross_variants({'before':baseline, 'after':changed})

    def test_unlogged_payload_mismatch_is_rejected(self):
        self.payload.write_bytes(b'wrong unlogged bytes')
        with self.assertRaisesRegex(ValueError, 'cache bytes differ'):
            analyze(self.directory, {})

    def test_different_performance_settings_are_allowed_with_explicit_warning(self):
        baseline = analyze(self.directory, {})
        for field, value in (
                ('threads', 8), ('delay_ms', 100.0), ('bandwidth_mib', 2.0),
                ('drain_timeout_s', 30.0), ('extra_args', ['--gui-reoptimize', '--quiet'])):
            with self.subTest(field=field):
                changed = copy.deepcopy(baseline)
                changed['provenance'][field] = value
                with self.assertWarnsRegex(RuntimeWarning, 'timings are not like-for-like') as warning:
                    validate_cross_variants({'before':baseline, 'after':changed})
                self.assertIn(field, str(warning.warning))

    def test_hidden_metadata_mismatch_is_rejected(self):
        self.metadata.write_bytes(b'wrong metadata bytes')
        with self.assertRaisesRegex(ValueError, 'cache bytes differ'):
            analyze(self.directory, {})

    def test_missing_request_log_and_legacy_drain_flag_are_explicit(self):
        del self.row['requests_drained']
        del self.configuration['drain_timeout_s']
        self.persist_recording()
        (self.directory/'cold-1.requests.json').unlink()
        summary = analyze(self.directory, {})
        self.assertEqual(summary['byte_verified_cache_objects'], 2)
        self.assertTrue(any('lack requests_drained' in note for note in summary['compatibility_notes']))
        self.assertTrue(any('request log is absent' in note for note in summary['compatibility_notes']))
        self.assertIsNone(summary['provenance']['drain_timeout_s'])
        self.assertTrue(any('request-drain timeout' in note for note in summary['compatibility_notes']))

    def test_failed_drain_is_rejected(self):
        self.row['requests_drained'] = False
        self.persist_recording()
        with self.assertRaisesRegex(ValueError, 'requests did not drain'):
            analyze(self.directory, {})

    def test_logged_payload_must_still_exist_in_verified_cache(self):
        write_json(self.directory/'cold-1.requests.json', [{
            'method':'GET', 'status':200, 'path':'/fiber/channel.zarr/0.0.0', 'bytes':16}])
        analyze(self.directory, {})
        self.payload.unlink()
        with self.assertRaisesRegex(ValueError, 'logged cache payload is missing'):
            analyze(self.directory, {})


class BenchmarkRequestAccountingTest(unittest.TestCase):
    def test_drain_timeout_accepts_only_positive_finite_seconds(self):
        for value in ('0', '-1', 'nan', 'inf', '-inf', 'invalid'):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_finite_seconds(value)
        self.assertEqual(positive_finite_seconds('30.5'), 30.5)
        self.assertEqual(positive_finite_seconds('5'), 5.0)

    def make_handler(self, path, root):
        handler = Handler.__new__(Handler)
        handler.server = SimpleNamespace(
            rows_condition=threading.Condition(), active_requests=0, rows=[], run_tag='cold-1',
            manifests={'fiber':root/'fiber.lasagna.json'}, delay=0, bytes_per_second=0)
        handler.path = path
        handler.command = 'GET'
        handler.wfile = io.BytesIO()
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        return handler

    def test_malformed_route_always_records_and_releases_request(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = self.make_handler('/fiber/%00', Path(directory))
            handler.serve(True)
            self.assertEqual(handler.server.active_requests, 0)
            self.assertEqual(len(handler.server.rows), 1)
            self.assertEqual(handler.server.rows[0]['status'], 400)
            self.assertEqual(handler.server.rows[0]['bytes'], 0)
            self.assertTrue(handler.close_connection)
            handler.send_header.assert_any_call('Content-Length', '0')

    def test_vanished_stat_target_always_records_and_releases_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'chunk').write_bytes(b'payload')
            handler = self.make_handler('/fiber/chunk', root)
            with mock.patch('benchmark_line_model_prefetch.os.fstat', side_effect=FileNotFoundError):
                handler.serve(True)
            self.assertEqual(handler.server.active_requests, 0)
            self.assertEqual(len(handler.server.rows), 1)
            self.assertEqual(handler.server.rows[0]['status'], 404)

    def test_failed_drain_result_is_preserved_and_rejected(self):
        condition = mock.MagicMock()
        condition.wait_for.return_value = False
        server = SimpleNamespace(rows_condition=condition, active_requests=1,
                                 rows=[{'previous':True}, {'current':True}])
        drained, requests = drain_trial_requests(server, 1, timeout=30.5)
        self.assertFalse(drained)
        self.assertEqual(condition.wait_for.call_args.kwargs['timeout'], 30.5)
        self.assertEqual(requests, [{'current':True}])
        with self.assertRaisesRegex(ValueError, 'requests did not drain'):
            require_drained_requests({'requests_drained':drained})
        require_drained_requests({'requests_drained':True})


if __name__ == '__main__':
    unittest.main()
