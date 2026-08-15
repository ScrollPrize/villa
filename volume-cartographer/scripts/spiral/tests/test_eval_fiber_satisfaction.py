import json

import numpy as np
import torch

import satisfaction_metrics


def test_eval_fibers_are_reported_separately(monkeypatch, tmp_path):
    monkeypatch.setenv('FIT_SPIRAL_SKIP_SAVE_MESH', '1')
    monkeypatch.setattr(
        satisfaction_metrics,
        'get_patch_satisfied_areas',
        lambda *args, **kwargs: (
            torch.zeros(0, dtype=torch.bool),
            torch.zeros(0),
            torch.zeros(0),
            [],
            torch.zeros(0, dtype=torch.bool),
            [],
        ),
    )
    monkeypatch.setattr(
        satisfaction_metrics,
        'get_unattached_pcl_satisfied_counts',
        lambda *args, **kwargs: (
            torch.tensor([2, 1]),
            torch.tensor([2, 2]),
            [torch.tensor([True, True]), torch.tensor([True, False])],
        ),
    )
    eval_fibers = [
        {'id': 10, 'name': 'held-out-a', 'source_file': 'eval_fibers/a.json',
         'zyxs': np.zeros((2, 3), dtype=np.float32),
         'windings': np.zeros(2, dtype=np.float32)},
        {'id': 11, 'name': 'held-out-b', 'source_file': 'eval_fibers/b.json',
         'zyxs': np.zeros((2, 3), dtype=np.float32),
         'windings': np.zeros(2, dtype=np.float32)},
    ]

    summary = satisfaction_metrics.save_overlay_and_print_satisfaction(
        'test',
        spiral_and_transform=None,
        slice_to_spiral_transform=None,
        dr_per_winding=torch.tensor(1.0),
        patches_list=[],
        patches_dict={},
        unattached_pcl_strips=[],
        eval_fiber_strips=eval_fibers,
        tracks=[],
        unverified_patches_list=[],
        unverified_patches_dict={},
        out_path=str(tmp_path),
        cfg={},
        z_begin=0,
        z_end=1,
        flow_field_radius=0,
        flow_min_corner_spiral_zyx=None,
        flow_max_corner_spiral_zyx=None,
        zs_for_visualisation=None,
        slice_yx=None,
        scroll_slices_for_visualisation=None,
        prediction_slices_for_visualisation=None,
        quad_label_map=None,
        z_to_umbilicus_yx=None,
        render_volume_scale=1,
        voxel_size_um=1.0,
        get_or_build_unattached_pcl_flat=lambda *args: None,
    )

    assert summary['satisfied_eval_fibers'] == 1
    assert summary['total_eval_fibers'] == 2
    assert summary['satisfied_eval_fiber_ratio'] == 0.5
    assert summary['satisfied_eval_fiber_points'] == 3
    assert summary['total_eval_fiber_points'] == 4
    assert summary['satisfied_eval_fiber_point_ratio'] == 0.75
    entries = json.loads((tmp_path / 'satisfied_test.json').read_text())
    assert [entry['name'] for entry in entries['eval_fibers']] == [
        'held-out-b', 'held-out-a']
    assert entries['eval_fibers'][0]['fraction'] == 0.5

