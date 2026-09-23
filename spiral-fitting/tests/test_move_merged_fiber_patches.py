import json

import pytest

from move_merged_fiber_patches import move_plan


def test_moves_only_success_patches_and_can_resume(tmp_path):
    source = tmp_path / 'graph_patches_only'
    source.mkdir()
    for name in ('merged_a', 'merged_b', 'shared_with_failed'):
        (source / name).mkdir()
        (source / name / 'marker').write_text(name)
    destination = tmp_path / 'graph_patches_merged'
    plan_path = tmp_path / 'fiber_merge_patch_move_plan.json'
    plan = {'source_root': str(source), 'destination_root': str(destination),
            'move_patches': ['merged_a', 'merged_b'],
            'keep_in_original': ['shared_with_failed'], 'move_count': 2, 'keep_count': 1}
    plan_path.write_text(json.dumps(plan))

    assert move_plan(plan_path, execute=False)['pending'] == 2
    assert not destination.exists()
    assert move_plan(plan_path, execute=True)['complete']
    assert move_plan(plan_path, execute=True)['already_moved'] == 2
    assert sorted(p.name for p in source.iterdir()) == ['shared_with_failed']
    assert sorted(p.name for p in destination.iterdir()) == ['merged_a', 'merged_b']
    assert (source / 'shared_with_failed' / 'marker').read_text() == 'shared_with_failed'

    plan['keep_in_original'] = []
    plan['keep_count'] = 0
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='changed after execution'):
        move_plan(plan_path, execute=True)
