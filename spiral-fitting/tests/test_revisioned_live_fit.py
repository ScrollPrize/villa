"""Opt-in CUDA fitting check; every writable input is copied under tmp_path."""
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from PIL import Image

import pytest
import torch

from fit_session import (SessionState, SpiralInputPaths, SpiralPreviewConfig,
                         SpiralRunConfig, parse_scroll_spec)
from input_publication import fingerprint
from spiral_runtime import create_session
from tifxyz import load_tifxyz


def make_real_revision_session(tmp_path):
    source = Path(os.environ['SPIRAL_REVISION_LIVE_DATASET'])
    patch_name = os.environ['SPIRAL_REVISION_PATCH']
    source_patch = source / 'verified_patches' / patch_name
    source_digest = fingerprint(source_patch)
    dataset = tmp_path / 'dataset'
    baseline = dataset / 'verified_patches' / 'baseline'
    shutil.copytree(source_patch, baseline)
    shutil.copyfile(source / 'umbilicus.json', dataset / 'umbilicus.json')
    metadata = json.loads((baseline / 'meta.json').read_text())
    metadata['spiral_patch_erode_cells'] = 0
    (baseline / 'meta.json').write_text(json.dumps(metadata))
    replacement = tmp_path / 'replacement'
    shutil.copytree(baseline, replacement)
    with Image.open(replacement / 'x.tif') as raster:
        values = np.array(raster)
    values[values != -1] += 2
    Image.fromarray(values).save(replacement / 'x.tif')
    patch = load_tifxyz(str(baseline))
    zs = patch.zyxs[..., 0][patch.valid_vertex_mask]
    spec = json.loads((source / 'spiral-scroll.json').read_text())
    config = {
        'dense_spacing_mode': 'grad_mag', 'loss_weight_dense_spacing': 0,
        'loss_weight_dense_normals': 0, 'loss_weight_shell_outer': 0,
        'loss_weight_shell_patch_radius': 0,
        'model_flow_voxel_resolution': 64,
        'sample_count_patches_per_step': 8, 'sample_count_patches_per_step_for_dt': 8,
        'sample_count_points_per_patch': 32, 'sample_count_regularisation_points': 64,
        'sample_count_shell_samples': 64,
        'sample_count_minimum_spacing_independent_samples': 64,
        'output_save_png_visualizations': False,
    }
    paths = SpiralInputPaths(dataset_root=str(dataset),
                             umbilicus=str(dataset / 'umbilicus.json'),
                             verified_patches=str(baseline.parent),
                             output_directory=str(tmp_path / 'output'),
                             cache_directory=str(tmp_path / 'cache'))
    session = create_session(paths, SpiralRunConfig(
        z_begin=int(zs.min()) - 1, z_end=int(zs.max()) + 2, config=config),
        SpiralPreviewConfig(), parse_scroll_spec(spec, dataset))
    return source_patch, source_digest, dataset, baseline, replacement, patch, config, session


@pytest.mark.skipif(not os.environ.get('SPIRAL_REVISION_LIVE_DATASET'),
                    reason='set SPIRAL_REVISION_LIVE_DATASET to opt into CUDA fitting')
def test_real_patch_revision_boundaries(tmp_path):
    source_patch, source_digest, dataset, baseline, replacement, patch, config, session = make_real_revision_session(tmp_path)
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
        record = {'id': logical_id, 'kind': 'patch',
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
            Path(report_path).write_text(
                json.dumps(report, indent=2))
