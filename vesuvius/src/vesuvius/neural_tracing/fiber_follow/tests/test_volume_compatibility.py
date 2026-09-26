"""Older volume metadata retains its defaults without weakening source checks."""
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.flow_matching import train
from vesuvius.neural_tracing.fiber_follow.shared.data import DATA_POLICY
from vesuvius.neural_tracing.fiber_follow.shared.experiment import read_manifest
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.flow_matching.model import ARCHITECTURE, FollowNetConfig
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


class StartupValidated(Exception):
    """Stop after metadata checks, before starting workers or opening volumes."""


def stop_after_validation(*args, **kwargs):
    raise StartupValidated


def write_manifest(path, volume):
    payload = dict(version=1, fibers=[], volume=volume, monitor=[], calibration=[], final=[],
                   monitor_fibers=[], calibration_fibers=[], final_fibers=[])
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(payload))
    return payload


@pytest.fixture
def legacy_manifest(tmp_path):
    spec = FiberVolumeSpec('unused', ct_zarr='unused', inputs='ct+presence')
    volume = spec.to_dict()
    volume.pop('load_presence')
    path = tmp_path/'seeds.json'
    payload = write_manifest(path, volume)
    return spec, path, payload


@pytest.mark.parametrize('mode', ['fresh', 'init', 'resume'])
@pytest.mark.parametrize('change', [None, {'load_presence': False}, {'ct_zarr': 'other'}, {'ct_grid_scale': 4.}])
def test_training_startup_with_legacy_volume_metadata(tmp_path, monkeypatch, legacy_manifest, mode, change):
    spec, manifest_path, manifest = legacy_manifest
    original_manifest = manifest_path.read_bytes()
    volume = dict(manifest['volume'], **(change or {}))
    if mode == 'fresh':
        write_manifest(manifest_path, volume)
    benchmark = tmp_path/'benchmark.json'
    benchmark.write_text(json.dumps(dict(architecture=ARCHITECTURE, passed=True, microbatch=2,
        flow_draws=64, crop=[176, 96, 96], sampler_mode='zero', compile_model=False,
        cache_training_encoding=True)))
    args = ['--device', 'cpu', '--fiber-zarrs', spec.fiber_zarr_dir, '--ct', spec.ct_zarr,
            '--fibers', 'unused', '--fixed-bank', 'unused', '--manifest', str(manifest_path),
            '--benchmark', str(benchmark), '--out-root', str(tmp_path/'output'), '--name', 'run',
            '--sampler-mode', 'zero', '--workers', '0']
    if mode != 'fresh':
        checkpoint_path = tmp_path/'source.pt'
        if mode == 'resume':
            run = tmp_path/'output'/'run'
            run.mkdir(parents=True)
            (run/'config.json').write_text('{}')
            checkpoint_path = run/'last.pt'
        model_cfg = FollowNetConfig().to_dict()
        model_cfg.pop('sampler_mode')  # Legacy deterministic checkpoints predate this field too.
        torch.save(dict(architecture=ARCHITECTURE, data_policy=DATA_POLICY, vol_spec=volume,
            model_cfg=model_cfg, n_history=128, seed_manifest_sha256=manifest['sha256'],
            crop=asdict(CropSpec(depth=176, width=96, behind=128, history_render='segments', history_sigma=.35))),
            checkpoint_path)
        args += ['--resume' if mode == 'resume' else '--init-from', str(checkpoint_path)]
    monkeypatch.setattr(train, 'load_fibers', lambda *a, **k: [])
    monkeypatch.setattr(train.OnPolicyStates, 'load', lambda path: SimpleNamespace())
    monkeypatch.setattr(train, 'OnlineCollector', stop_after_validation)
    expected = {'fresh': 'Frozen manifest', 'init': 'Initialization volume', 'resume': 'Resumed draws/volume'}
    if change:
        with pytest.raises(ValueError, match=expected[mode]):
            train.main(args)
    else:
        with pytest.raises(StartupValidated):
            train.main(args)
        assert manifest_path.read_bytes() == original_manifest
        assert read_manifest(manifest_path)['sha256'] == manifest['sha256']


@pytest.mark.parametrize('change', [None, {'load_presence': False}, {'ct_zarr': 'other'}, {'ct_grid_scale': 4.}])
def test_evaluation_accepts_legacy_defaults_but_rejects_changed_volume(tmp_path, monkeypatch, legacy_manifest, change):
    spec, manifest_path, _ = legacy_manifest
    path = Path(__file__).parents[1]/'scripts'/'evaluate_single_path.py'
    module_spec = importlib.util.spec_from_file_location('eval_volume_compatibility', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'load_fibers', stop_after_validation)
    loader = lambda *a: (None, None, 128, replace(spec, **(change or {})), {})
    args = ['calibrate', '--device', 'cpu', '--checkpoints', 'unused', '--thresholds', '.5',
            '--manifest', str(manifest_path), '--out', str(tmp_path/'evaluation')]
    if change:
        with pytest.raises(ValueError, match='Volume differs'):
            module.main(args, checkpoint_loader=loader)
    else:
        with pytest.raises(StartupValidated):
            module.main(args, checkpoint_loader=loader)
