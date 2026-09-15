"""Opt-in CUDA fitting check; every writable input is copied under tmp_path."""

from live_revision_fixtures import make_real_revision_session
import json
import os
from pathlib import Path
import time
import uuid

import pytest
import torch

from fit_session import SessionState
from input_publication import fingerprint
from tifxyz import load_tifxyz


@pytest.mark.skipif(not os.environ.get('SPIRAL_REVISION_LIVE_DATASET'),
                    reason='set SPIRAL_REVISION_LIVE_DATASET to opt into CUDA fitting')
@pytest.mark.parametrize("influence", [False, True])
def test_real_patch_revision_boundaries(tmp_path, influence):
    source_patch, source_digest, dataset, baseline, replacement, patch, config, session = make_real_revision_session(tmp_path, influence)
    report = {'source': str(source_patch), 'config': config, 'boundaries': []}

    def wait_idle():
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status = session.status()
            assert status['state'] != SessionState.Error, status
            if status['state'] == SessionState.Idle:
                return
            time.sleep(.02)
        raise AssertionError(session.status())

    def apply(records):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        batch_id = str(uuid.uuid4())
        try:
            result = session.apply_input_changes(batch_id, records, timeout=120)
        except TimeoutError:
            result = session.apply_input_changes(batch_id, records, timeout=180)
        report['boundaries'].append({
            'seconds': time.perf_counter() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'operation': 'delete' if records[0].get('deleted') else 'replace', 'result': result})
        return result

    try:
        wait_idle()
        context = session._context
        model, optimiser = context.spiral_and_transform, context.optimiser
        logical_id = str(uuid.uuid4())
        record = {'id': logical_id, 'kind': 'patch', 'role': 'verified',
                  'source_id': 'baseline', 'path': str(replacement), 'revision': 2}
        session.run(2, autosave_on_pause=False)
        wait_idle()
        model_before = [p.detach().clone() for group in optimiser.param_groups
                        for p in group['params']]
        completed = session.completed_iterations
        assert apply([record])['applied']
        assert session.completed_iterations == completed
        assert context.spiral_and_transform is model and context.optimiser is optimiser
        for before, after in zip(model_before, [p for g in optimiser.param_groups for p in g['params']]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        assert list(context.verified_patches) == ['baseline']
        torch.testing.assert_close(context.verified_patches['baseline'].zyxs,
                                   load_tifxyz(str(replacement)).zyxs, rtol=0, atol=0)
        # Invalid selected batch never removes its preceding valid member.
        invalid = {**record, 'id': str(uuid.uuid4()), 'path': str(tmp_path / 'missing')}
        assert not apply([{**record, 'deleted': True, 'revision': 3}, invalid])['applied']
        assert list(context.verified_patches) == ['baseline']
        session.run(12, autosave_on_pause=False)
        assert apply([{**record, 'deleted': True, 'revision': 3}])['applied']
        assert not context.verified_patches
        assert apply([{**record, 'revision': 4}])['applied']
        wait_idle()
        assert list(context.verified_patches) == ['baseline']
        assert context.spiral_and_transform is model and context.optimiser is optimiser
        vertices = patch.zyxs[patch.valid_vertex_mask]
        endpoints = vertices[[0, -1]][:, [2, 1, 0]].tolist()
        pcl_path = tmp_path / 'same_windings.json'
        pcl_path.write_text(json.dumps({'vc_pointcollections_json_version': '1',
            'collections': {'7': {'name': 'live-draft', 'points': {
                str(index): {'p': xyz} for index, xyz in enumerate(endpoints)}}}}))
        fiber_path = tmp_path / 'fiber.json'
        fiber_path.write_text(json.dumps({'type': 'vc3d_fiber', 'line_points': [],
            'control_points': [[value * 4 for value in xyz] for xyz in endpoints]}))
        pcl_id, fiber_id = str(uuid.uuid4()), str(uuid.uuid4())
        mixed = [
            {'id': pcl_id, 'kind': 'pcl', 'role': 'same_winding', 'source_id': '7',
             'source_path': str(pcl_path), 'path': str(pcl_path), 'revision': 1},
            {'id': fiber_id, 'kind': 'fiber', 'source_id': 'fiber',
             'path': str(fiber_path), 'revision': 1}]
        session.run(4, autosave_on_pause=False)
        assert apply(mixed)['applied']
        assert fiber_id in context.fiber_catalog
        assert apply([{**item, 'deleted': True, 'revision': 2} for item in mixed])['applied']
        assert fiber_id not in context.fiber_catalog
        assert apply([{**item, 'revision': 3} for item in mixed])['applied']
        wait_idle()
        resident_pcl = context._workspace_membership[pcl_id]['resident_id']
        session.run(1, run_config={'input_use_pcl_same_winding': False}, autosave_on_pause=False)
        wait_idle()
        assert resident_pcl not in context.regular_pcl_catalog
        assert resident_pcl in context._source_point_collections
        session.run(1, run_config={'input_use_pcl_same_winding': True}, autosave_on_pause=False)
        wait_idle()
        assert resident_pcl in context.regular_pcl_catalog
        assert context.regular_pcl_catalog[resident_pcl]['metadata']['logical_input_id'] == pcl_id
        # Repeat warm boundaries, retaining the trained model across runs.
        for revision in (5, 8, 11):
            session.run(4, autosave_on_pause=False)
            assert apply([{**record, 'revision': revision}])['applied']
            assert apply([{**record, 'deleted': True, 'revision': revision + 1}])['applied']
            assert apply([{**record, 'revision': revision + 2}])['applied']
            wait_idle()
        report['completed_iterations'] = session.completed_iterations
        report['device'] = torch.cuda.get_device_name()
        assert fingerprint(source_patch) == source_digest
    finally:
        report["final_status"] = session.status()
        session.close(timeout=30)
        report_path = os.environ.get('SPIRAL_REVISION_LIVE_REPORT')
        if report_path:
            Path(report_path).with_suffix(f'.influence-{int(influence)}.json').write_text(
                json.dumps(report, indent=2))
