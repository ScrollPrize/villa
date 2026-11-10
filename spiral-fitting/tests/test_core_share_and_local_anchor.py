"""patch_sampling_core_share reserves draws for core patches; abs_winding_local_to_attachment looks up the attachment ij."""
import types

import numpy as np

import fit_spiral
import losses


def _patch(area, weight, core):
    return types.SimpleNamespace(_sampling_area=area, area=area, loss_weight_scale=weight,
                                 inner_core_radius=(100.0 if core else None))


def _probabilities(share, patches, overrides=None):
    owner = types.SimpleNamespace(config={'patch_sampling_area_exponent': 1.0, 'patch_sampling_core_share': share,
                                          'patch_loss_weight_scale_overrides': overrides})
    return fit_spiral.FitContext._patch_sampling_probabilities(owner, patches)


def test_loss_weight_scale_overrides_match_the_longest_entry_name_prefix():
    a = _patch(1.0, 1.0, False); a.entry_name = 'strip-auto-7'
    b = _patch(1.0, 1.0, False); b.entry_name = 'strip-auto-v2-7'
    c = _patch(1.0, 1.0, False); c.entry_name = 'band-3'
    d = _patch(1.0, 1.0, False)  # no entry name (interactive strip): untouched
    p = _probabilities(0.0, [a, b, c, d], {'strip-auto-': 0.5, 'strip-auto-v2-': 2.0})
    assert np.allclose(p, np.array([0.5, 2.0, 1.0, 1.0]) / 4.5)
    assert np.allclose(_probabilities(0.0, [a, b, c, d], None), 0.25)


def test_fiber_sampling_group_separates_auto_fibers_only_when_enabled():
    auto_tag = {'metadata': {'fiber_tags': ['auto']}, 'file_basename': 'kb_1.json'}
    auto_name = {'metadata': {'fiber_tags': []}, 'file_basename': 'auto_00000012.json'}
    manual = {'metadata': {'fiber_tags': []}, 'file_basename': 'kb_1.json'}
    on = {'pcl_auto_fiber_sampling_group': True}
    assert fit_spiral.fiber_sampling_group(auto_tag, on) == 'auto_fibers'
    assert fit_spiral.fiber_sampling_group(auto_name, on) == 'auto_fibers'
    assert fit_spiral.fiber_sampling_group(manual, on) == 'fibers'
    assert fit_spiral.fiber_sampling_group(auto_tag, {'pcl_auto_fiber_sampling_group': False}) == 'fibers'


def test_core_share_zero_keeps_area_weight_draws():
    p = _probabilities(0.0, [_patch(1.0, 1.0, False), _patch(3.0, 1.0, True)])
    assert np.allclose(p, [0.25, 0.75])


def test_core_share_reserves_draws_and_keeps_zero_weights_zero():
    p = _probabilities(0.25, [_patch(9.0, 1.0, False), _patch(1.0, 1.0, True), _patch(5.0, 0.0, True)])
    # base draws 0.9 / 0.1 / 0; core-only draws 0 / 1 / 0
    assert np.allclose(p, [0.75 * 0.9, 0.75 * 0.1 + 0.25, 0.0])
    assert np.isclose(p.sum(), 1.0)


def test_core_share_without_core_patches_is_a_no_op():
    p = _probabilities(0.25, [_patch(1.0, 1.0, False), _patch(1.0, 1.0, False)])
    assert np.allclose(p, [0.5, 0.5])


def test_annotation_lookup_ij_floors_onto_the_annotation_quad():
    patches = {'p': types.SimpleNamespace(_sampling_valid_quad_mask_np=np.ones((4, 6), dtype=bool))}
    assert losses._annotation_lookup_ij(patches, 'p', 1.25, 2.75) == (1.25, 2.75)
    i, j = losses._annotation_lookup_ij(patches, 'p', 7.5, -2.0)  # clamped to the last / first quad
    assert (int(i), int(j)) == (3, 0) and 3 <= i < 4 and 0 <= j < 1


def _exclusions(config, patches, collections):
    owner = types.SimpleNamespace(config=config, verified_patches=patches)
    return fit_spiral.FitContext._fiber_link_exclusions(owner, collections)


def test_fiber_link_exclusions_keep_each_fiber_off_its_own_strip_and_apply_the_regex():
    patches = {'strip-net-1': types.SimpleNamespace(source_fiber='kb_1.json'),
               'strip-v2b-7': types.SimpleNamespace(source_fiber='auto_v00000007.json'),
               'band-3': types.SimpleNamespace(source_fiber=None)}
    collections = {1: {'file_basename': 'kb_1.json'}, 2: {'source_file': '/x/auto_v00000007.json'},
                   3: {'file_basename': 'kb_9.json'}}
    assert _exclusions({}, patches, collections) == (frozenset(), {})
    own = _exclusions({'pcl_fiber_link_exclude_own_strip': True}, patches, collections)
    assert own == (frozenset(), {1: frozenset({'strip-net-1'}), 2: frozenset({'strip-v2b-7'})})
    regex = _exclusions({'pcl_fiber_link_exclude_patch_regex': '^strip-'}, patches, collections)
    assert regex == (frozenset({'strip-net-1', 'strip-v2b-7'}), {})
    both = _exclusions({'pcl_fiber_link_exclude_patch_regex': 'net', 'pcl_fiber_link_exclude_own_strip': True},
                       patches, collections)
    assert both == (frozenset({'strip-net-1'}), {2: frozenset({'strip-v2b-7'})})


def test_patch_payload_round_trips_source_fiber():
    import torch
    import tifxyz
    zyxs = torch.tensor([[[0., 0., 0.], [0., 0., 1.]], [[0., 1., 0.], [0., 1., 1.]]])
    patch = tifxyz.Patch(zyxs, torch.tensor([1.0, 1.0]), None, None, 'u', None, 0.3, None, 'kb_1.json')
    again = tifxyz.patch_from_payload(tifxyz.patch_to_payload(patch))
    assert again.source_fiber == 'kb_1.json' and again.loss_weight_scale == 0.3
    assert tifxyz.patch_from_payload({**tifxyz.patch_to_payload(patch), 'source_fiber': None}).source_fiber is None
