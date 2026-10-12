import pytest
import torch

import devices
from spiral_helpers import _segmented_median_per_strip


@pytest.fixture(autouse=True)
def _fresh_precision_setting():
    # max_precision_bits() reads its environment variable once per process.
    devices.max_precision_bits.cache_clear()
    yield
    devices.max_precision_bits.cache_clear()


def test_device_order_is_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.delenv('FIT_SPIRAL_DEVICE', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert devices.fit_device() == torch.device('cuda')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert devices.fit_device() == torch.device('mps')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert devices.fit_device() == torch.device('cpu')


def test_device_override(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')
    assert devices.fit_device() == torch.device('cpu')
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'tpu')
    with pytest.raises(ValueError, match='unsupported device'):
        devices.fit_device()
    assert devices.fit_device('cpu') == torch.device('cpu')
    with pytest.raises(RuntimeError, match='CUDA is not available'):
        devices.fit_device('cuda')


def test_float_hi_is_float64_except_on_mps(monkeypatch):
    monkeypatch.delenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', raising=False)
    assert devices.float_hi(torch.device('cuda')) == torch.float64
    assert devices.float_hi('cpu') == torch.float64
    assert devices.float_hi() == torch.float64
    assert devices.float_hi('mps') == torch.float32
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '32')
    devices.max_precision_bits.cache_clear()
    assert devices.float_hi('cuda') == torch.float32
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '16')
    devices.max_precision_bits.cache_clear()
    with pytest.raises(ValueError):
        devices.float_hi('cpu')


def test_precision_setting_is_checked_when_the_device_is_chosen(monkeypatch):
    # A bad value fails at startup, where the fit picks its device, not at
    # the first optimizer step after the inputs have loaded.
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '16')
    with pytest.raises(ValueError, match='must be 32 or 64'):
        devices.fit_device('cpu')
    # Read once: later changes to the variable do not reach a running fit.
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '32')
    devices.max_precision_bits.cache_clear()
    assert devices.float_hi('cpu') == torch.float32
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '64')
    assert devices.float_hi('cpu') == torch.float32


def _strip_context(device):
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, 40, (3000,), generator=generator)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    strip_id = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    # Offsets of whole windings plus sub-voxel jitter: values the float32
    # composite key (strip_id * range + value) could no longer tell apart.
    values = (torch.randint(0, 60, (int(starts[-1]),), generator=generator).float()
              + torch.rand(int(starts[-1]), generator=generator) * 1e-3)
    return {
        'normalised_radii': values.to(device), 'strip_id': strip_id.to(device),
        'starts': starts.to(device), 'lengths': lengths.to(device),
        'S': len(lengths), 'device': torch.device(device),
    }


def _reference_medians(ctx):
    medians = []
    for start, length in zip(ctx['starts'][:-1].tolist(), ctx['lengths'].tolist()):
        strip = ctx['normalised_radii'][start:start + length].cpu().sort().values
        medians.append(strip[(length - 1) // 2])
    return torch.stack(medians)


def test_both_segmented_median_sorts_match_per_strip_sort(monkeypatch):
    # The float64 composite key (CPU, CUDA) and the two-pass sort (MPS, or
    # FIT_SPIRAL_MAX_PRECISION_FLOAT=32) on the same strips.
    ctx = _strip_context('cpu')
    expected = _reference_medians(ctx)
    monkeypatch.delenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', raising=False)
    assert torch.equal(_segmented_median_per_strip(ctx), expected)
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '32')
    devices.max_precision_bits.cache_clear()
    assert torch.equal(_segmented_median_per_strip(ctx), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS')
def test_segmented_median_on_mps(monkeypatch):
    monkeypatch.delenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', raising=False)
    ctx = _strip_context('mps')
    assert torch.equal(_segmented_median_per_strip(ctx).cpu(), _reference_medians(ctx))



@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS')
def test_patch_atlas_upload_on_mps_survives_freed_staging(monkeypatch):
    # An atlas larger than the staging bound is uploaded in several flushes,
    # each from a host tensor that is freed while the next one is built. With
    # torch 2.11 and 2.12 a non_blocking host-to-MPS copy can read its source
    # after that (pytorch/pytorch#189690), so earlier flushes arrive corrupted.
    import types

    import numpy as np

    import fit_spiral

    monkeypatch.setattr(fit_spiral, 'PATCH_ATLAS_STAGING_BYTES', 1 << 20)
    generator = torch.Generator().manual_seed(0)
    patches = {
        key: types.SimpleNamespace(
            zyxs=torch.rand(256, 128, 3, generator=generator) * 100.,
            _sampling_valid_quad_mask_np=np.ones((255, 127), dtype=bool),
            _sampling_2d_path=None)
        for key in range(24)
    }
    expected = torch.cat([p.zyxs.reshape(-1, 3) for p in patches.values()])
    for _ in range(3):
        atlas = fit_spiral.PatchAtlas(patches, device='mps').materialize()
        torch.mps.synchronize()
        assert torch.equal(atlas.zyxs_flat.cpu(), expected)


@pytest.mark.parametrize('override,backend', [(None, 'nccl'), ('cpu', 'gloo')])
def test_ddp_backend_follows_the_fit_device(monkeypatch, override, backend):
    import ddp_helpers
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    if override is None:
        monkeypatch.delenv('FIT_SPIRAL_DEVICE', raising=False)
    else:
        monkeypatch.setenv('FIT_SPIRAL_DEVICE', override)
    calls = {}
    monkeypatch.setattr(torch.cuda, 'set_device', lambda rank: calls.setdefault('set_device', rank))
    monkeypatch.setattr(ddp_helpers.dist, 'init_process_group',
                        lambda **kwargs: calls.setdefault('backend', kwargs['backend']))
    monkeypatch.setattr(ddp_helpers.dist, 'barrier', lambda **kwargs: None)
    try:
        ddp_helpers.maybe_init_distributed(ddp_helpers.DistributedContext(rank=0, world_size=2))
    finally:
        ddp_helpers.set_process_context(ddp_helpers.SINGLE_PROCESS)
    assert calls['backend'] == backend
    assert ('set_device' in calls) == (backend == 'nccl')


def test_prefetcher_is_used_only_for_a_cuda_patch_atlas(monkeypatch):
    # FIT_SPIRAL_PREFETCH=1 with FIT_SPIRAL_DEVICE=cpu on a CUDA machine: the
    # prefetcher's CUDA streams must not be built for CPU tensors.
    import types
    import numpy as np
    import losses
    import prefetch
    from fit_spiral import PatchAtlas
    from theta_crossing_map import ThetaCrossingMap
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setenv('FIT_SPIRAL_PREFETCH', '1')

    def no_prefetcher():
        raise AssertionError('prefetcher used for a non-CUDA patch atlas')
    monkeypatch.setattr(prefetch, 'get_prefetcher', no_prefetcher)

    def fake_patch(seed):
        generator = torch.Generator().manual_seed(seed)
        return types.SimpleNamespace(
            zyxs=torch.rand(8, 8, 3, generator=generator) * 100.,
            _sampling_valid_quad_mask_np=np.ones((7, 7), dtype=bool),
            _sampling_2d_path=None)
    patches = {f'p{n}': fake_patch(30 + n) for n in range(3)}
    atlas = PatchAtlas(patches, device='cpu').materialize()
    crossing_map = ThetaCrossingMap('cpu')
    atlas.register_theta_topology(crossing_map)
    crossing_map.force_refresh(lambda value: value)
    _, _, zyxs, _, mask = losses._sample_patch_batch(
        'cpu_atlas', list(patches.values()), np.full(3, 1 / 3), num_to_sample=4,
        point_cap=6, cfg={}, patch_atlas=atlas, crossing_map=crossing_map)
    assert tuple(zyxs.shape) == (4, 6, 3) and zyxs.device.type == 'cpu'
    assert bool(mask.all())


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_ink_metric_folds_take_turns_off_cuda(monkeypatch, tmp_path, device):
    # Off CUDA every fold would share one device, so they must not all start
    # at once, and the workers get no CUDA_VISIBLE_DEVICES.
    import numpy as np
    from PIL import Image
    import get_ink_metrics
    if device == 'mps':
        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    ink = tmp_path / 'ink'
    ink.mkdir()
    Image.fromarray(np.zeros((8, 16), np.uint8)).save(ink / 'w000-005_flat.jpg')
    model = tmp_path / 'model'
    for fold in range(3):
        (model / f'fold_{fold}').mkdir(parents=True)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    launched = []

    class FakeWorker:
        def __init__(self, cmd, env, **kwargs):
            assert all(worker.done for worker in launched), 'two folds ran at once'
            launched.append(self)
            self.cmd, self.env, self.done = cmd, env, False

        def wait(self):
            self.done = True
            return 0

    class Stop(Exception):
        pass

    def stop(*args):
        raise Stop
    monkeypatch.setattr(get_ink_metrics.subprocess, 'Popen', FakeWorker)
    monkeypatch.setattr(get_ink_metrics, 'load_fold_prob', stop)
    with pytest.raises(Stop):
        get_ink_metrics.run_main([str(ink), '--model', str(model), '--device', device,
                                  '--output', str(tmp_path / 'out')])
    assert len(launched) == 3
    for worker in launched:
        assert 'CUDA_VISIBLE_DEVICES' not in worker.env
        assert worker.cmd[worker.cmd.index('--device') + 1] == device


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('mps', marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason='needs MPS'))])
def test_smoothing_kernel_agrees_with_the_startup_report(monkeypatch, device):
    # Near sigma ~ 0.17 the kernel's centre weight crosses the 1 - 1e-7
    # "is it a delta" test; a float32 build of the kernel decides some of
    # these sigmas differently from the float64 startup report.
    from flow_grad_smoothing import gaussian_kernel
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '32')
    devices.max_precision_bits.cache_clear()
    for sigma in torch.linspace(0.170, 0.176, 601, dtype=torch.float64).tolist():
        report = gaussian_kernel(sigma, dtype=torch.float64)
        fit = gaussian_kernel(sigma, torch.device(device), torch.float32)
        assert (report is None) == (fit is None), sigma
        if fit is not None:
            assert torch.equal(fit.cpu(), report.float())


@pytest.mark.parametrize('device,built_on', [
    ('cuda', 'cuda'), ('cpu', 'cpu'),
    pytest.param('mps', 'cpu', marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason='needs MPS'))])
def test_smoothing_kernel_is_built_on_its_device_except_mps(monkeypatch, device, built_on):
    # CUDA keeps building the float64 kernel on the GPU; only MPS, which has
    # no float64, builds it on the CPU. The recorder makes every tensor on
    # the CPU, so the CUDA case runs without CUDA.
    import flow_grad_smoothing
    seen = []
    arange = torch.arange

    def recording_arange(*args, device=None, **kwargs):
        seen.append(None if device is None else torch.device(device).type)
        return arange(*args, **kwargs)
    monkeypatch.setattr(flow_grad_smoothing.torch, 'arange', recording_arange)
    kernel = flow_grad_smoothing.gaussian_kernel(2.0, torch.device(device))
    assert seen == [built_on]
    assert kernel is not None and kernel.dtype == torch.float32


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('mps', marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason='needs MPS'))])
def test_step_timer_off_cuda(monkeypatch, capsys, device):
    # FIT_SPIRAL_PROFILE_STEPS=1 with the fit on the CPU or MPS.
    from ddp_helpers import StepTimer
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', device)
    timer = StepTimer(enabled=True, report=True, window=1)
    x = torch.randn(64, 64, device=device)
    timer.start('fwd')
    (x @ x).sum()
    timer.stop('fwd')
    # Nothing queued in between, as for 'comm' in a single-process fit.
    timer.start('comm')
    timer.stop('comm')
    timer.tick()
    timer.maybe_report(1)
    out = capsys.readouterr().out
    assert '[profile step 1] avg/step over 1: fwd=' in out and 'comm=' in out


def test_get_spiral_points_off_cuda(monkeypatch):
    # get_spiral_points leaves the device to get_spiral_yxs, whose default
    # was 'cuda'.
    import numpy as np
    from sample_spiral import get_spiral_points, get_spiral_yxs
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')
    yxs = get_spiral_points(np.zeros((64, 64)), np.array([32., 32.]))
    assert len(yxs) and ((yxs >= 0) & (yxs < 64)).all()
    assert get_spiral_yxs(3, 10, 4).device == torch.device('cpu')


def test_fit_device_takes_a_torch_device_or_an_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert devices.fit_device(torch.device('cuda', 0)) == torch.device('cuda', 0)
    assert devices.fit_device('cuda:1') == torch.device('cuda', 1)
    assert devices.fit_device(torch.device('cpu')) == torch.device('cpu')
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'CUDA:0')
    assert devices.fit_device() == torch.device('cuda', 0)
    with pytest.raises(ValueError, match='unsupported device'):
        devices.fit_device('xla')


def test_distributed_fit_on_mps_stops_before_joining(monkeypatch):
    # torch.distributed has no MPS backend: gloo raises NotImplementedError at
    # the first all_reduce of an MPS gradient, after the inputs have loaded.
    import ddp_helpers
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'mps')

    def no_join(**kwargs):
        raise AssertionError('joined a process group for an MPS fit')
    monkeypatch.setattr(ddp_helpers.dist, 'init_process_group', no_join)
    monkeypatch.setattr(ddp_helpers.dist, 'barrier', lambda **kwargs: None)
    try:
        with pytest.raises(RuntimeError, match='cannot run on MPS'):
            ddp_helpers.maybe_init_distributed(ddp_helpers.DistributedContext(rank=0, world_size=2))
    finally:
        ddp_helpers.set_process_context(ddp_helpers.SINGLE_PROCESS)
