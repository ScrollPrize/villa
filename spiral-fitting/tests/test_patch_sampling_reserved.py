"""patch_sampling_reserved (frozen/s1fh, PLAN-fencehold-v2 C3): a fixed share of draws for a named patch group."""
import os
import types

import numpy as np
import pytest

import config
import fit_spiral

S1FH = '/media/djosey/nvme4/spiral_0332/frozen/s1fh/'


def test_imports_resolve_under_s1fh():
    assert os.path.realpath(fit_spiral.__file__).startswith(S1FH)
    assert os.path.realpath(config.__file__).startswith(S1FH)


def _patch(name, area, weight=1.0, holdout=False):
    return types.SimpleNamespace(_sampling_area=area, area=area, loss_weight_scale=weight,
                                 inner_core_radius=None, entry_name=name, holdout=holdout)


def _probabilities(patches, reserved=None, core_share=0.0, exponent=0.5, holdout=True):
    owner = types.SimpleNamespace(config={
        'patch_sampling_area_exponent': exponent, 'patch_sampling_core_share': core_share,
        'patch_loss_weight_scale_overrides': None, 'exclude_holdout_patches_from_training': holdout,
        'patch_sampling_reserved': reserved})
    return fit_spiral.FitContext._patch_sampling_probabilities(owner, patches)


PREFIXES = ['strip-net', 'strip-man', 'strip-cap']


def _family(n_manual, n_auto, rng):
    manual = [_patch(f'{PREFIXES[i % 3]}-{i}', float(rng.uniform(1e3, 1e5))) for i in range(n_manual)]
    auto = [_patch(f'strip-auto-{i}', float(rng.uniform(1e3, 1e5))) for i in range(n_auto)]
    return manual, auto


@pytest.mark.parametrize('n_auto', [3000, 30])   # the other family 10x and 0.1x the reserved group
def test_share_is_exact_whatever_the_size_of_the_other_family(n_auto):
    rng = np.random.default_rng(0)
    manual, auto = _family(300, n_auto, rng)
    p = _probabilities(manual + auto, {'prefixes': PREFIXES, 'share': 0.15})
    assert abs(p[:300].sum() - 0.15) < 1e-9
    assert abs(p.sum() - 1.0) < 1e-9


def test_relative_weights_within_each_group_are_kept():
    rng = np.random.default_rng(1)
    manual, auto = _family(20, 50, rng)
    patches = manual + auto
    base = _probabilities(patches, None)
    p = _probabilities(patches, {'prefixes': PREFIXES, 'share': 0.3})
    assert np.allclose(p[:20] / p[:20].sum(), base[:20] / base[:20].sum(), rtol=1e-6)
    assert np.allclose(p[20:] / p[20:].sum(), base[20:] / base[20:].sum(), rtol=1e-6)


def test_holdout_and_zero_weights_stay_zero():
    patches = [_patch('strip-net-1', 10.0), _patch('strip-net-2', 10.0, weight=0.0),
               _patch('band-1', 10.0, holdout=True), _patch('band-2', 10.0), _patch('strip-auto-1', 40.0)]
    p = _probabilities(patches, {'prefixes': ['strip-net'], 'share': 0.2})
    assert p[1] == 0.0 and p[2] == 0.0
    assert abs(p[0] - 0.2) < 1e-12
    assert abs(p[3] + p[4] - 0.8) < 1e-12


def test_off_reproduces_the_original_probabilities_bit_for_bit():
    rng = np.random.default_rng(2)
    manual, auto = _family(40, 400, rng)
    patches = manual + auto
    owner = types.SimpleNamespace(config={
        'patch_sampling_area_exponent': 0.5, 'patch_sampling_core_share': 0.0,
        'patch_loss_weight_scale_overrides': None, 'exclude_holdout_patches_from_training': True})
    legacy = fit_spiral.FitContext._patch_sampling_probabilities(owner, patches)   # key absent
    off = _probabilities(patches, None)
    assert off.dtype == legacy.dtype and off.tobytes() == legacy.tobytes()
    assert config.Config().patch_sampling_reserved is None


def test_runtime_errors():
    patches = [_patch('strip-net-1', 10.0), _patch('strip-auto-1', 10.0)]
    with pytest.raises(ValueError, match='not in'):
        _probabilities(patches, {'prefixes': ['strip-net'], 'share': 1.0})
    with pytest.raises(ValueError, match='not in'):
        _probabilities(patches, {'prefixes': ['strip-net'], 'share': -0.1})
    with pytest.raises(ValueError, match='no patch with prefix'):
        _probabilities(patches, {'prefixes': ['strip-cap'], 'share': 0.1})
    with pytest.raises(ValueError, match='no patch with prefix'):
        _probabilities([_patch('strip-net-1', 10.0, weight=0.0), _patch('strip-auto-1', 10.0)],
                       {'prefixes': ['strip-net'], 'share': 0.1})
    with pytest.raises(ValueError, match='no non-reserved'):
        _probabilities([_patch('strip-net-1', 10.0), _patch('band-1', 10.0, holdout=True)],
                       {'prefixes': ['strip-net'], 'share': 0.1})
    with pytest.raises(ValueError, match='core_share'):
        _probabilities(patches, {'prefixes': ['strip-net'], 'share': 0.1}, core_share=0.2)


def test_config_validation():
    ok = config.Config({'patch_sampling_reserved': {'prefixes': PREFIXES, 'share': 0.15}})
    assert ok.patch_sampling_reserved['share'] == 0.15
    for bad in ({'prefixes': [], 'share': 0.1}, {'prefixes': ['strip-net'], 'share': 1.0},
                {'prefixes': ['strip-net']}, {'prefixes': 'strip-net', 'share': 0.1},
                {'prefixes': ['strip-net'], 'share': 0.1, 'extra': 1}, {'prefixes': [''], 'share': 0.1}):
        with pytest.raises(ValueError):
            config.Config({'patch_sampling_reserved': bad})
    with pytest.raises(ValueError, match='core_share'):
        config.Config({'patch_sampling_reserved': {'prefixes': ['strip-net'], 'share': 0.1},
                       'patch_sampling_core_share': 0.2})


def test_key_is_a_checkpoint_supervision_and_new_fit_key():
    assert 'patch_sampling_reserved' in config.CHECKPOINT_SUPERVISION_KEYS
    assert 'patch_sampling_reserved' in config.NEW_FIT_KEYS


def test_logs_the_realized_share_once(capsys):
    patches = [_patch('strip-net-1', 10.0), _patch('strip-auto-1', 30.0)]
    owner = types.SimpleNamespace(config={
        'patch_sampling_area_exponent': 1.0, 'patch_sampling_core_share': 0.0,
        'patch_loss_weight_scale_overrides': None, 'exclude_holdout_patches_from_training': True,
        'patch_sampling_reserved': {'prefixes': ['strip-net'], 'share': 0.25}})
    fit_spiral.FitContext._patch_sampling_probabilities(owner, patches)
    fit_spiral.FitContext._patch_sampling_probabilities(owner, patches)
    out = capsys.readouterr().out
    assert out.count('PATCH_SAMPLING_RESERVED') == 1
    assert 'realized_reserved_mass=0.250000000000' in out


def test_empty_pool_raises_with_reservation_and_returns_none_without():
    # astra fh2 F5: a configured reservation never silently disappears with an empty active pool
    with pytest.raises(ValueError, match='active patch pool is empty'):
        _probabilities([], reserved={'prefixes': PREFIXES, 'share': 0.15})
    assert _probabilities([], reserved=None) is None


def test_all_patches_filtered_to_zero_weight_raises_with_reservation():
    rng = np.random.default_rng(3)
    manual, auto = _family(5, 50, rng)
    for p in manual + auto:
        p.holdout = True
    with pytest.raises(ValueError):
        _probabilities(manual + auto, reserved={'prefixes': PREFIXES, 'share': 0.15})


def test_rejection_rebuild_guard_present():
    # the rejection rebuild (fit_spiral, after dropping rejected patches) bypasses _patch_sampling_probabilities when the list is
    # empty; with a reservation configured it must raise instead (source check: the rebuild needs a full FitContext to exercise)
    import inspect
    src = inspect.getsource(fit_spiral)
    assert "patch_sampling_reserved is set but every patch was rejected" in src
