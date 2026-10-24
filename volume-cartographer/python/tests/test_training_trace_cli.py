"""Opt-in parity with the existing training benchmark on cached real predictions.

This is CPU tracing only: no model or GPU inference is launched on a cache miss.
"""
import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.mark.parametrize('extra', [[], ['--replay-reverse-interval-base', '128',
    '--replay-reverse-normal-base', '20', '--replay-reverse-tangential-base', '40',
    '--replay-reverse-to-start', '--replay-reverse-all-endpoints'],
    ['--replay-presence-cutoff', '0.75', '--replay-presence-mode', 'modulated']])
def test_existing_training_replay_cli_parity(tmp_path, extra):
    keys = ('VC_TRAIN_TRACE_BINARY', 'VC_TRAIN_TRACE_BASELINE', 'VC_TEST_WIDTH_MANIFEST',
            'VC_TEST_WIDTH_FIBER', 'VC_TEST_REVERSE_NORMAL_MANIFEST')
    if not all(os.environ.get(key) for key in keys):
        pytest.skip('set VC_TRAIN_TRACE_BINARY/BASELINE and real width/fiber/normal fixtures')
    binary, baseline, manifest, fiber_path, normal_manifest = (Path(os.environ[key]) for key in keys)
    annotation = json.loads(fiber_path.read_text())
    points = np.asarray(annotation['line_points'])
    start = int(np.argmin(np.linalg.norm(points - annotation['control_points'][0]['position'], axis=1)))
    points = points[start:]
    arc = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    count = min(len(points), int(np.searchsorted(arc, 256)) + 1)
    annotation['line_points'] = points[:count].tolist()
    annotation['control_points'] = [dict(annotation['control_points'][0], position=points[0].tolist()),
                                    dict(position=points[count-1].tolist())]
    if 'baked_annotations' in annotation:
        for key in ('normals_xyz', 'valid', 'widths_base'):
            if key in annotation['baked_annotations']:
                annotation['baked_annotations'][key] = annotation['baked_annotations'][key][start:start+count]
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(annotation))
    settings = dict(version=1, backend='beam', workers=2, decoded_cache_bytes=256 << 20,
        error_threshold_base_voxels=40, tangential_threshold_factor=2,
        excursion_grace_base_voxels=160, match_refine_steps=1,
        trace_args=['--threads', '1', '--beam-width', '8', '--beam-lookahead-steps', '2',
                    '--step-voxels', '6', '--width-relative-tolerance', '0.4',
                    '--width-weighting', 'cosine', *extra],
        sources=[dict(id='real-paris4', base_voxel_size_um=2.4,
            fiber_manifest=str(manifest), normal_manifest=str(normal_manifest),
            reference_normals='baked', tracing_normals='fiber', inference_scaledown_power=0,
            coordinate_space=annotation['vc_open_data_coordinate_space'], fibers=[str(path)])])
    documents = []
    for name, executable in [('baseline', baseline), ('extracted', binary)]:
        config = copy.deepcopy(settings)
        config['output'] = str(tmp_path / name)
        file = tmp_path / (name + '.json')
        file.write_text(json.dumps(config))
        subprocess.run([str(executable), 'replay-set', '--config', str(file)], check=True, timeout=120)
        documents.append([json.loads((tmp_path / name / 'cases' / str(i) / 'trace.json').read_text())
                          for i in range(2)])
    for before, after in zip(*documents):
        assert after['reference_points_base_xyz'] == before['reference_points_base_xyz']
        before_replay, after_replay = before['replay'], after['replay']
        # Accumulated float costs have cross-build roundoff; thresholds below are
        # for the independently built historical executable, not the runtime objective.
        # Committed geometry, indices, termination/filter outcomes remain exact.
        for left, right in zip(before_replay['segments'], after_replay['segments'], strict=True):
            assert right['trace_points_base_xyz'] == left['trace_points_base_xyz']
            np.testing.assert_allclose(right.pop('trace_cumulative_losses'),
                                       left.pop('trace_cumulative_losses'), rtol=1e-5, atol=1e-6)
        def compare(left, right):
            if isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    compare(left[key], right[key])
            elif isinstance(left, list):
                assert len(left) == len(right)
                for a, b in zip(left, right):
                    compare(a, b)
            elif isinstance(left, float):
                np.testing.assert_allclose(right, left, rtol=1e-12, atol=1e-12)
            else:
                assert right == left
        compare(before_replay, after_replay)
    print('REAL_TRAIN_TRACE_PARITY', extra or 'unfiltered', 'two directions; exact geometry/outcomes; historical float-cost roundoff bounded')
