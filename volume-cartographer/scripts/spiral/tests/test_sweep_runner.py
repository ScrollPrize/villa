import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest


SWEEP_DIR = Path(__file__).resolve().parents[1] / 'sweep'
sys.path.insert(0, str(SWEEP_DIR))

import sweep_fit_spiral as sweep
import sweep_run_wrapper as wrapper


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def _legacy_result(root, run_name, root_config, seed_config, seed, *,
                   column_score, satisfaction_score, completed_ns):
    run_dir = root / run_name
    seed_dir = run_dir / f'seed_{seed}'
    fit_dir = seed_dir / 'fit'
    metrics_path = fit_dir / 'meshes' / 'fitted' / 'ink_metric' / 'metrics.json'
    _write_json(run_dir / 'config.json', root_config)
    _write_json(seed_dir / 'config.json', seed_config)
    _write_json(fit_dir / 'satisfaction_summary.json', {
        'satisfied_patch_ratio': satisfaction_score,
    })
    _write_json(metrics_path, {'summary': {
        'total_fg_pixels': 10,
        'total_pixels': 100,
        'overall_fg_fraction': 0.1,
        'overall_line_score': 0.2,
        'overall_column_score': column_score,
    }})
    os.utime(metrics_path, ns=(completed_ns, completed_ns))
    return metrics_path


def _legacy_satisfaction(root, run_name, root_config, seed_config, seed, score):
    run_dir = root / run_name
    seed_dir = run_dir / f'seed_{seed}'
    _write_json(run_dir / 'config.json', root_config)
    _write_json(seed_dir / 'config.json', seed_config)
    _write_json(seed_dir / 'fit' / 'satisfaction_summary.json', {
        'satisfied_patch_ratio': score,
    })


def test_aggregate_logs_only_means_and_sample_std():
    payload = wrapper.aggregate_across_seeds({
        1: {'score': 1.0, 'only_seed_one': 5.0},
        2: {'score': 3.0},
        3: {'score': 5.0},
    }, 'final')

    assert payload == {
        'final/score': 3.0,
        'final/score_std': 2.0,
    }
    assert not any(key.startswith('seed') for key in payload)


def test_seed_fit_environment_disables_wandb_and_removes_identity(tmp_path):
    inherited = {
        'WANDB_MODE': 'online',
        'WANDB_RUN_ID': 'parent',
        'WANDB_NAME': 'parent-name',
        'WANDB_RUN_GROUP': 'group',
        'WANDB_RESUME': 'allow',
        'WANDB_SWEEP_ID': 'new-sweep',
        'WANDB_SWEEP_PARAM_PATH': '/tmp/params.yaml',
        'CUDA_VISIBLE_DEVICES': '0,1',
    }

    env = wrapper.seed_fit_environment(
        inherited, {'optimizer_random_seed': 2}, tmp_path / 'seed_2')

    assert env['WANDB_MODE'] == 'disabled'
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert all(var not in env for var in wrapper.SEED_WANDB_IDENTITY_VARS)
    assert json.loads(env['FIT_SPIRAL_CONFIG_OVERRIDES']) == {
        'optimizer_random_seed': 2,
    }


def test_reuse_selects_newest_exact_config_per_seed(tmp_path):
    root = tmp_path / 'old-sweep'
    root.mkdir()
    root_config = wrapper.Config({'z_begin': 5000}).as_dict()
    seed_config = wrapper.Config({
        'z_begin': 5000,
        'optimizer_random_seed': 1,
    }).as_dict()
    older = _legacy_result(
        root, 'older', root_config, seed_config, 1,
        column_score=0.2, satisfaction_score=0.3, completed_ns=1_000_000_000)
    newer = _legacy_result(
        root, 'newer', root_config, seed_config, 1,
        column_score=0.8, satisfaction_score=0.9, completed_ns=2_000_000_000)
    # A similar-looking but non-identical resolved config must never be reused.
    wrong_config = wrapper.Config({'z_begin': 6000}).as_dict()
    _legacy_result(
        root, 'wrong-config', wrong_config, seed_config, 1,
        column_score=1.0, satisfaction_score=1.0, completed_ns=3_000_000_000)

    selected = wrapper.select_reusable_seed(
        [root], root_config, 1, seed_config, require_ink=True)

    assert selected['source'] == str(newer.resolve())
    assert selected['ink']['overall_column_score'] == 0.8
    assert selected['final']['satisfied_patch_ratio'] == 0.9
    assert selected['candidate_sources'] == [str(older.resolve()), str(newer.resolve())]
    assert wrapper.select_reusable_seed(
        [root], root_config, 2,
        wrapper.Config({'z_begin': 5000, 'optimizer_random_seed': 2}).as_dict(),
        require_ink=True) is None


def test_new_result_manifest_is_reusable_without_legacy_outputs(tmp_path):
    root = tmp_path / 'old-sweep'
    run_dir = root / 'aggregate-run'
    root_config = wrapper.Config({'z_begin': 7000}).as_dict()
    seed_config = wrapper.Config({
        'z_begin': 7000,
        'optimizer_random_seed': 3,
    }).as_dict()
    wrapper.write_json_atomic(run_dir / 'config.json', root_config)
    wrapper.write_json_atomic(run_dir / 'seed_3' / 'config.json', seed_config)
    wrapper.write_json_atomic(run_dir / 'seed_3' / 'result.json',
                              wrapper.make_seed_result(
                                  3, seed_config,
                                  {'satisfied_patch_ratio': 0.4},
                                  {'overall_column_score': 0.5},
                                  {'kind': 'computed'}))

    selected = wrapper.select_reusable_seed(
        [root], root_config, 3, seed_config, require_ink=True)

    assert selected['source_kind'] == 'result_manifest'
    assert selected['final'] == {'satisfied_patch_ratio': 0.4}
    assert selected['ink'] == {'overall_column_score': 0.5}


def test_combined_objective_is_aggregated_after_per_seed_calculation():
    final = {1: {'s': 1.0}, 2: {'s': 3.0}, 3: {'s': 5.0}}
    ink = {1: {'c': 2.0}, 2: {'c': 4.0}, 3: {'c': 6.0}}
    objective = {'final/s': 2.0, 'ink/c': 0.5}

    values = [wrapper.combined_objective(objective, seed, final, ink)
              for seed in (1, 2, 3)]

    assert values == [3.0, 8.0, 13.0]
    assert sum(values) / len(values) == 8.0
    assert math.sqrt(sum((value - 8.0) ** 2 for value in values) / 2) == 5.0


def test_wrapper_command_propagates_reuse_roots(tmp_path):
    first = tmp_path / 'first-old-sweep'
    second = tmp_path / 'second-old-sweep'
    args = Namespace(
        dataset=str(tmp_path / 'dataset'),
        out_root=str(tmp_path / 'out'),
        scroll_spec=None,
        cache=None,
        base_config=None,
        reuse_root=[str(first), str(second)],
    )

    command = sweep.build_wrapper_command({
        'seeds': [1, 2, 3],
        'eval': {'skip_eval': True},
    }, args)

    pairs = [(command[index], command[index + 1])
             for index in range(len(command) - 1)
             if command[index] == '--reuse-root']
    assert pairs == [
        ('--reuse-root', str(first.resolve())),
        ('--reuse-root', str(second.resolve())),
    ]


def test_run_creates_then_launches_returned_new_sweep_id(monkeypatch):
    calls = []
    args = object()
    monkeypatch.setattr(sweep, 'cmd_create',
                        lambda actual_args: calls.append(('create', actual_args)) or 'new123')
    monkeypatch.setattr(sweep, 'cmd_launch',
                        lambda actual_args, sweep_id: calls.append(
                            ('launch', actual_args, sweep_id)))

    sweep.cmd_run(args)

    assert calls == [
        ('create', args),
        ('launch', args, 'new123'),
    ]


def test_all_reused_seeds_produce_one_aggregate_log_and_manifest(
        tmp_path, monkeypatch):
    reuse_root = tmp_path / 'old-sweep'
    root_config = wrapper.Config({'z_begin': 5000}).as_dict()
    for seed, score in ((1, 0.2), (2, 0.4), (3, 0.6)):
        seed_config = wrapper.Config({
            'z_begin': 5000,
            'optimizer_random_seed': seed,
        }).as_dict()
        _legacy_satisfaction(
            reuse_root, 'old-run', root_config, seed_config, seed, score)

    logged = []
    monkeypatch.setattr(wrapper, 'log_to_run',
                        lambda payload, files=(): logged.append((payload, list(files))))
    monkeypatch.setenv('WANDB_RUN_ID', 'new-run')
    monkeypatch.setenv('WANDB_SWEEP_ID', 'new-sweep')
    monkeypatch.setattr(sys, 'argv', [
        'sweep_run_wrapper.py',
        '--dataset', str(tmp_path / 'dataset'),
        '--params-json', '{"z_begin": 5000}',
        '--out-root', str(tmp_path / 'new-output'),
        '--reuse-root', str(reuse_root),
        '--skip-eval',
    ])

    wrapper.main()

    assert len(logged) == 1
    payload, files = logged[0]
    assert payload['final/satisfied_patch_ratio'] == pytest.approx(0.4)
    assert payload['final/satisfied_patch_ratio_std'] == pytest.approx(0.2)
    assert not any(key.startswith('seed') for key in payload)
    aggregate_path = (tmp_path / 'new-output' / 'new-sweep' / 'new-run'
                      / 'aggregate_results.json')
    assert files == [aggregate_path]
    manifest = json.loads(aggregate_path.read_text())
    assert manifest['sweep_id'] == 'new-sweep'
    assert manifest['run_id'] == 'new-run'
    assert all(seed['provenance']['kind'] == 'reused'
               for seed in manifest['per_seed'].values())


def test_invalid_reuse_root_is_reported_by_wrapper_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'sweep_run_wrapper.py',
        '--dataset', str(tmp_path / 'dataset'),
        '--params-json', '{}',
        '--skip-eval',
        '--reuse-root', str(tmp_path / 'missing'),
    ])

    with pytest.raises(SystemExit, match='2'):
        wrapper.main()
