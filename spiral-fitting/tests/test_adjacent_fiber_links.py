"""Adjacent fiber links (format v4 ``adjacent_branches``) behind pcl_use_adjacent_fiber_links.

An adjacent link ties a vertical fiber to the horizontal fiber one winding
outside it (W_V = W_H - 1). Off, the fitter ignores them exactly as before;
on, they join the link components with a step of one winding carried by the
members' winding annotations.
"""
import json
import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch

import losses
from config import NEW_FIT_KEYS, Config, FitConfig
from fit_spiral import (
    FitContext,
    _UnattachedPclStripList,
    materialize_fiber_fit_inputs,
)
from spiral_helpers import (
    build_link_components,
    link_component_winding_offsets,
    load_fiber_point_collection,
    load_fiber_point_collections,
    resolve_fiber_links,
)


def _record(local_index, target_id, target_index, pending=False):
    return {
        'local_index': local_index,
        'branch_file': f'{target_id}.json',
        'branch_index': target_index,
        'pending': pending,
    }


def _member(cid, logical_id, tag, *, branches=(), adjacent=(), x_offset=0.0,
            attached=True, diagonal=False):
    """A two-point resident fiber collection as materialize_fiber_fit_inputs sees it.

    ``tag`` is VC3D's manual H/V tag (None leaves the geometric fallback to
    decide: the points run along x, which classifies as H, unless
    ``diagonal`` runs them along (z, x) equally, which is ambiguous and stays
    untagged)."""
    points = {}
    for point_id in range(2):
        x = x_offset + point_id
        z = x if diagonal else 0.0
        points[point_id] = {
            'id': point_id,
            'collectionId': cid,
            'p': [x, 0.0, z],
            'zyx': np.asarray([z, 0.0, x], dtype=np.float32),
            'winding_annotation': float('nan'),
            **({'on_patch': {'id': f'patch-{logical_id}'}} if attached else {}),
        }
    return {
        'id': cid,
        'name': logical_id,
        'file_basename': f'{logical_id}.json',
        'sampling_group': 'fibers',
        'metadata': {
            'logical_input_id': logical_id,
            'logical_input_kind': 'fiber',
            'winding_is_absolute': False,
            'hv_classification': {'manual_tag': tag} if tag else {},
        },
        'points': points,
        'kept_orig_indices': np.asarray([0, 1]),
        'control_line_indices': np.asarray([0, 1]),
        'branches': [_record(*entry) for entry in branches],
        'adjacent_branches': [_record(*entry) for entry in adjacent],
    }


def _manual_tags(catalog):
    def hv_tag_of(cid):
        for pcl in catalog.values():
            if pcl['id'] == cid:
                return pcl['metadata']['hv_classification'].get('manual_tag')
        raise KeyError(cid)
    return hv_tag_of


def _materialize(catalog, **kwargs):
    patches = {f'patch-{logical_id}': object() for logical_id in catalog}
    return materialize_fiber_fit_inputs(
        catalog, patches, z_begin=-10, z_end=10, z_margin=0,
        min_point_spacing=0, **kwargs)


def _annotations(pcl):
    return [point['winding_annotation'] for _, point in
            sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))]


def _h_adjacent_v(order='hv'):
    """H fiber ``a`` and V fiber ``b`` joined by one reciprocal adjacent link."""
    a = _member(10, 'a', 'H', adjacent=[(1, 'b', 0)], x_offset=0)
    b = _member(11, 'b', 'V', adjacent=[(0, 'a', 1)], x_offset=10)
    return {'a': a, 'b': b} if order == 'hv' else {'b': b, 'a': a}


# --- configuration -----------------------------------------------------------

def test_flag_defaults_off_and_needs_a_new_fit():
    assert Config().pcl_use_adjacent_fiber_links is False
    assert FitConfig(Config().as_dict())['pcl_use_adjacent_fiber_links'] is False
    assert 'pcl_use_adjacent_fiber_links' in NEW_FIT_KEYS


# --- loader --------------------------------------------------------------------

def _write_fiber(directory, name, document):
    path = os.path.join(directory, name)
    with open(path, 'w') as stream:
        json.dump({'type': 'vc3d_fiber', **document}, stream)
    return path


def test_loader_parses_adjacent_records_but_only_keeps_their_endpoints_when_enabled(capsys):
    # Scaled x = 0, 10, 30: a 15 vx spacing drops the middle point unless it
    # is an adjacent-link endpoint that the fit will use.
    document = {
        'control_points': [[0, 0, 0], [40, 0, 0], [120, 0, 0]],
        'line_points': [[0, 0, 0], [40, 0, 0], [120, 0, 0]],
        'adjacent_branches': [{
            'branch_file': '/somewhere/peer.json',
            'control_point_index': 1,
            'branch_control_point_index': 0,
            'pending': True,
        }],
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = _write_fiber(temporary, 'fiber.json', document)

        off = load_fiber_point_collection(path, 7, min_point_spacing=15)
        assert 'ignoring 1 adjacent fiber link(s)' in capsys.readouterr().out
        assert off['branches'] == []
        assert off['adjacent_branches'] == [
            {'local_index': 1, 'branch_file': 'peer.json',
             'branch_index': 0, 'pending': True}]
        assert off['kept_orig_indices'].tolist() == [0, 2]

        on = load_fiber_point_collection(
            path, 7, min_point_spacing=15, use_adjacent_links=True)
        assert 'ignoring' not in capsys.readouterr().out
        assert on['branches'] == []
        assert on['adjacent_branches'] == off['adjacent_branches']
        assert on['kept_orig_indices'].tolist() == [0, 1, 2]


# --- resolver ----------------------------------------------------------------

def test_resolver_ignores_adjacent_links_unless_asked():
    catalog = _h_adjacent_v()
    collections = {pcl['id']: pcl for pcl in catalog.values()}
    assert resolve_fiber_links(collections) == []
    assert resolve_fiber_links(collections, include_pending=True) == []


def test_resolver_requires_a_tag_source_when_adjacent_links_are_requested():
    collections = {pcl['id']: pcl for pcl in _h_adjacent_v().values()}
    with pytest.raises(ValueError):
        resolve_fiber_links(collections, include_adjacent=True)


@pytest.mark.parametrize('order, expected', [('hv', -1), ('vh', 1)])
def test_adjacent_link_resolves_once_with_the_vertical_side_one_winding_inside(order, expected):
    catalog = _h_adjacent_v(order)
    collections = {pcl['id']: pcl for pcl in catalog.values()}
    links = resolve_fiber_links(
        collections, include_adjacent=True, hv_tag_of=_manual_tags(catalog))
    # Both files carry the reciprocal record; it resolves to one link whose
    # delta is the winding of the b endpoint relative to the a endpoint.
    assert len(links) == 1
    link, = links
    first = next(iter(catalog.values()))
    assert link['a_coll'] == first['id']
    assert link['delta'] == expected
    assert link['pending'] is False
    assert {link['a_point'], link['b_point']} == {1, 0}


def test_ordinary_links_carry_a_zero_delta():
    a = _member(10, 'a', 'H', branches=[(1, 'b', 0)])
    b = _member(11, 'b', 'V', branches=[(0, 'a', 1)], x_offset=10)
    links = resolve_fiber_links({10: a, 11: b})
    assert [link['delta'] for link in links] == [0]


@pytest.mark.parametrize('tags', [('H', 'H'), ('V', 'V'), (None, 'V'), ('H', None)])
def test_adjacent_link_without_one_h_and_one_v_contributes_no_constraint(tags, capsys):
    tag_a, tag_b = tags
    a = _member(10, 'a', tag_a, adjacent=[(1, 'b', 0)], diagonal=tag_a is None)
    b = _member(11, 'b', tag_b, adjacent=[(0, 'a', 1)], x_offset=10,
                diagonal=tag_b is None)
    catalog = {'a': a, 'b': b}

    links = resolve_fiber_links(
        {10: a, 11: b}, include_adjacent=True, hv_tag_of=_manual_tags(catalog))

    # Dropped outright: never demoted to a same-winding (delta 0) link.
    assert links == []
    out = capsys.readouterr().out
    assert 'dropping adjacent fiber link' in out
    assert 'contributes no winding constraint' in out

    # The materialized views agree: no component, every annotation 0.
    cross, strips, _, resolved, components = _materialize(
        catalog, use_adjacent_links=True)
    assert resolved == [] and components == []
    assert len(cross) == 2
    assert all(_annotations(pcl) == [0.0, 0.0] for pcl in catalog.values())
    assert strips == []


def test_records_disagreeing_on_the_link_kind_drop_the_link(capsys):
    # a stores the pair as an ordinary link, b as an adjacent one.
    a = _member(10, 'a', 'H', branches=[(1, 'b', 0)])
    b = _member(11, 'b', 'V', adjacent=[(0, 'a', 1)], x_offset=10)
    catalog = {'a': a, 'b': b}

    links = resolve_fiber_links(
        {10: a, 11: b}, include_adjacent=True, hv_tag_of=_manual_tags(catalog))

    assert links == []
    assert 'disagree on the link kind' in capsys.readouterr().out
    # Without adjacent links the ordinary record alone still resolves.
    assert len(resolve_fiber_links({10: a, 11: b})) == 1


def test_untagged_geometry_is_classified_by_the_materializer_thresholds():
    # No VC3D tags at all: the geometric fallback tags the x-aligned fiber H
    # and the z-aligned one V, so the adjacent link resolves through
    # materialize_fiber_fit_inputs' own hv_tag_of.
    a = _member(10, 'a', None, adjacent=[(1, 'b', 0)])
    b = _member(11, 'b', None, adjacent=[(0, 'a', 1)], x_offset=10)
    for point in b['points'].values():
        z = float(point['zyx'][2]) - 10.0
        point['zyx'] = np.asarray([z, 0.0, 10.0], dtype=np.float32)
        point['p'] = [10.0, 0.0, z]
    catalog = {'a': a, 'b': b}

    _, _, _, resolved, _ = _materialize(catalog, use_adjacent_links=True)

    assert [link['delta'] for link in resolved] == [-1]
    assert _annotations(a) == [0.0, 0.0]
    assert _annotations(b) == [-1.0, -1.0]


# --- component offsets -------------------------------------------------------

def test_offsets_compose_along_h_adjacent_v_same_winding_h2():
    a = _member(10, 'a', 'H', adjacent=[(1, 'b', 0)])
    b = _member(11, 'b', 'V', adjacent=[(0, 'a', 1)], branches=[(1, 'c', 0)],
                x_offset=10)
    c = _member(12, 'c', 'H', branches=[(0, 'b', 1)], x_offset=20)
    catalog = {'a': a, 'b': b, 'c': c}
    links = resolve_fiber_links(
        {10: a, 11: b, 12: c}, include_adjacent=True,
        hv_tag_of=_manual_tags(catalog))
    assert sorted(link['delta'] for link in links) == [-1, 0]
    (member_cids, member_links), = build_link_components(links)

    offsets, inconsistent = link_component_winding_offsets(
        member_cids, member_links)

    assert inconsistent == []
    assert offsets == {10: 0, 11: -1, 12: -1}


def test_offsets_start_from_the_first_member():
    a = _member(10, 'a', 'H', adjacent=[(1, 'b', 0)])
    b = _member(11, 'b', 'V', adjacent=[(0, 'a', 1)], x_offset=10)
    links = resolve_fiber_links(
        {10: a, 11: b}, include_adjacent=True, hv_tag_of=_manual_tags({'a': a, 'b': b}))
    assert link_component_winding_offsets([11, 10], links)[0] == {11: 0, 10: 1}
    assert link_component_winding_offsets([10, 11], links)[0] == {10: 0, 11: -1}


def test_inconsistent_link_cycle_is_reported_and_the_tree_assignment_stands():
    # a~b adjacent (-1), b-c ordinary (0), c-a ordinary (0): the cycle sums
    # to -1, so one link must contradict the others.
    a = _member(10, 'a', 'H', adjacent=[(1, 'b', 0)], branches=[(0, 'c', 1)])
    b = _member(11, 'b', 'V', adjacent=[(0, 'a', 1)], branches=[(1, 'c', 0)],
                x_offset=10)
    c = _member(12, 'c', 'H', branches=[(0, 'b', 1), (1, 'a', 0)], x_offset=20)
    catalog = {'a': a, 'b': b, 'c': c}
    links = resolve_fiber_links(
        {10: a, 11: b, 12: c}, include_adjacent=True,
        hv_tag_of=_manual_tags(catalog))
    assert len(links) == 3
    (member_cids, member_links), = build_link_components(links)

    offsets, inconsistent = link_component_winding_offsets(
        member_cids, member_links)

    assert len(inconsistent) == 1
    assert set(offsets) == {10, 11, 12}
    for link in member_links:
        if link in inconsistent:
            continue
        assert offsets[link['b_coll']] - offsets[link['a_coll']] == link['delta']


# --- materialized views ------------------------------------------------------

def test_flag_off_leaves_adjacent_links_out_of_every_view():
    a = _member(10, 'a', 'H', adjacent=[(1, 'b', 0)])
    b = _member(11, 'b', 'V', adjacent=[(0, 'a', 1)], branches=[(1, 'c', 0)],
                x_offset=10)
    c = _member(12, 'c', 'H', branches=[(0, 'b', 1)], x_offset=20)
    catalog = {'a': a, 'b': b, 'c': c}

    cross, strips, _, resolved, components = _materialize(catalog)

    assert [link['delta'] for link in resolved] == [0]
    assert [sorted(member_cids) for member_cids, _ in components] == [[11, 12]]
    assert len(cross) == 2  # a alone, plus the b-c component
    merged = next(pcl for pcl in cross if 'link_member_cids' in pcl)
    assert 'link_member_winding_offsets' not in merged
    assert all(_annotations(pcl) == [0.0, 0.0] for pcl in catalog.values())
    assert all(np.all(strip['windings'] == 0.0) for strip in strips)
    assert [strip['id'] for strip in strips] == [11, 12]


def test_merged_component_carries_the_one_winding_step_across_the_junction():
    a = _member(10, 'a', 'H', adjacent=[(1, 'b', 0)])
    b = _member(11, 'b', 'V', adjacent=[(0, 'a', 1)], branches=[(1, 'c', 0)],
                x_offset=10)
    c = _member(12, 'c', 'H', branches=[(0, 'b', 1)], x_offset=20)
    catalog = {'a': a, 'b': b, 'c': c}

    cross, strips, _, resolved, components = _materialize(
        catalog, use_adjacent_links=True)

    assert sorted(link['delta'] for link in resolved) == [-1, 0]
    assert [member_cids for member_cids, _ in components] == [[10, 11, 12]]
    merged, = cross
    assert merged['link_member_cids'] == [10, 11, 12]
    assert merged['link_member_winding_offsets'] == {10: 0, 11: -1, 12: -1}
    # The shared point dictionaries carry the offsets ...
    assert _annotations(a) == [0.0, 0.0]
    assert _annotations(b) == [-1.0, -1.0]
    assert _annotations(c) == [-1.0, -1.0]
    # ... so the relative-winding loss's winding_diff across the junction is
    # -1 from H to V (+1 the other way) and 0 across the ordinary link, and
    # the component chain routes through both junctions.
    h_point, v_point, h2_point = a['points'][1], b['points'][0], c['points'][0]
    chain = merged['chain']
    assert v_point['winding_annotation'] - h_point['winding_annotation'] == -1
    assert h_point['winding_annotation'] - v_point['winding_annotation'] == 1
    assert h2_point['winding_annotation'] - v_point['winding_annotation'] == 0
    assert h2_point['winding_annotation'] - h_point['winding_annotation'] == -1
    assert [id(p) for p in chain.points_between(h_point, h2_point)] == [
        id(a['points'][1]), id(b['points'][0]), id(b['points'][1]), id(c['points'][0])]
    # The strips (unattached view) copy the same per-point windings and keep
    # the junction points.
    strips_by_id = {strip['id']: strip for strip in strips}
    assert strips_by_id[10]['windings'].tolist() == [0.0, 0.0]
    assert strips_by_id[11]['windings'].tolist() == [-1.0, -1.0]
    assert strips_by_id[12]['windings'].tolist() == [-1.0, -1.0]
    assert strips_by_id[10]['link_points'] == {1: 1}
    assert strips_by_id[11]['link_points'] == {0: 0, 1: 1}


def test_rematerializing_without_the_flag_resets_the_shared_offsets():
    catalog = _h_adjacent_v()
    _materialize(catalog, use_adjacent_links=True)
    assert _annotations(catalog['b']) == [-1.0, -1.0]

    cross, _, _, resolved, _ = _materialize(catalog)

    assert resolved == []
    assert len(cross) == 2
    assert _annotations(catalog['a']) == [0.0, 0.0]
    assert _annotations(catalog['b']) == [0.0, 0.0]


def test_unattached_component_edges_hop_adjacent_junctions_with_the_step_in_the_windings():
    catalog = _h_adjacent_v()
    _, strips, groups, resolved, components = _materialize(
        catalog, use_adjacent_links=True)
    context = FitContext.__new__(FitContext)
    context.unattached_pcl_strips = _UnattachedPclStripList(list(strips))
    context.unattached_strip_sampling_groups = list(groups)
    context.link_components = list(components)
    context.unattached_components = []
    context.unattached_component_groups = []
    context.unattached_component_edges = []

    context._rebuild_unattached_components()

    assert context.unattached_components == [[0, 1]]
    assert context.unattached_component_groups == ['fibers']
    (strip_a, pos_a, strip_b, pos_b), = context.unattached_component_edges[0]
    windings_a = strips[strip_a]['windings']
    windings_b = strips[strip_b]['windings']
    # Walking H -> V across this junction steps one winding inward.
    assert windings_b[pos_b] - windings_a[pos_a] == -1.0


# --- end to end through the loader and the relative-winding loss --------------

def test_loaded_fibers_resolve_adjacent_links_from_vc3d_automatic_tags():
    def fiber(points_xyz, tag, certainty, peer, local_index, peer_index):
        return {
            'control_points': points_xyz,
            'line_points': points_xyz,
            'hv_classification': {'manual_tag': '', 'automatic_tag': tag,
                                  'automatic_certainty': certainty},
            'adjacent_branches': [{
                'branch_file': peer, 'control_point_index': local_index,
                'branch_control_point_index': peer_index, 'pending': True}],
        }
    horizontal = [[0, 0, 0], [100, 0, 0], [200, 0, 0]]
    vertical = [[100, 0, -100], [100, 0, 0], [100, 0, 100]]
    with tempfile.TemporaryDirectory() as temporary:
        _write_fiber(temporary, 'h.json', fiber(horizontal, 'H', 0.9, 'v.json', 1, 1))
        _write_fiber(temporary, 'v.json', fiber(vertical, 'V', 0.9, 'h.json', 1, 1))
        # Below the certainty threshold the geometric fallback decides (x-aligned:
        # H), and a V fiber whose automatic tag disagrees with its geometry
        # follows the confident automatic tag.
        _write_fiber(temporary, 'h2.json', fiber(horizontal, 'H', 0.1, 'v2.json', 1, 1))
        _write_fiber(temporary, 'v2.json', fiber(horizontal, 'V', 0.9, 'h2.json', 1, 1))
        # Two confident horizontals: no constraint.
        _write_fiber(temporary, 'h3.json', fiber(horizontal, 'H', 0.9, 'h4.json', 1, 1))
        _write_fiber(temporary, 'h4.json', fiber(horizontal, 'H', 0.9, 'h3.json', 1, 1))

        loaded, _ = load_fiber_point_collections(
            temporary, 1, min_point_spacing=0, use_adjacent_links=True)
        catalog = {}
        for pcl in loaded.values():
            logical_id = os.path.splitext(pcl['file_basename'])[0]
            pcl['metadata'].update(logical_input_kind='fiber', logical_input_id=logical_id)
            catalog[logical_id] = pcl

    for pcl in catalog.values():
        assert pcl['branches'] == [] and len(pcl['adjacent_branches']) == 1

    _, strips, _, resolved, components = materialize_fiber_fit_inputs(
        catalog, {}, z_begin=-1000, z_end=1000, z_margin=0, min_point_spacing=0,
        use_pending_links=True, use_adjacent_links=True)

    by_pair = {frozenset((link['a_coll'], link['b_coll'])): link['delta']
               for link in resolved}
    ids = {logical_id: pcl['id'] for logical_id, pcl in catalog.items()}
    assert by_pair == {
        frozenset((ids['h'], ids['v'])): -1,
        frozenset((ids['h2'], ids['v2'])): -1,
    }
    strips_by_id = {strip['id']: strip for strip in strips}
    assert set(strips_by_id) == set(ids.values())
    for name in ('h', 'h2', 'h3', 'h4'):
        assert strips_by_id[ids[name]]['windings'].tolist() == [0.0, 0.0, 0.0]
    for name in ('v', 'v2'):
        assert strips_by_id[ids[name]]['windings'].tolist() == [-1.0, -1.0, -1.0]
    # Pending adjacent links follow the pending switch like ordinary ones.
    _, _, _, resolved, _ = materialize_fiber_fit_inputs(
        catalog, {}, z_begin=-1000, z_end=1000, z_margin=0, min_point_spacing=0,
        use_adjacent_links=True)
    assert resolved == []


def test_rel_winding_loss_demands_one_winding_step_across_an_adjacent_link():
    from test_winding_pcl_unwrap import DR, _IdentityTransform, _cfg, _setup, _spiral_point

    # H fiber a on raw winding 5 attached to p1; V fiber b one winding inside
    # (raw winding 4) attached to p2; adjacent link a[1] ~ b[0].
    a = _member(10, 'a', 'H', adjacent=[(1, 'b', 0)])
    b = _member(11, 'b', 'V', adjacent=[(0, 'a', 1)])
    coordinates = {
        (10, 0): _spiral_point(0.1, 5), (10, 1): _spiral_point(0.2, 5),
        (11, 0): _spiral_point(0.2, 4), (11, 1): _spiral_point(0.3, 4),
    }
    node_ids = {}
    refs = []
    for pcl, patch_id in ((a, 'p1'), (b, 'p2')):
        for point_id, point in pcl['points'].items():
            zyx = coordinates[(pcl['id'], point_id)]
            point['zyx'] = np.asarray(zyx, dtype=np.float32)
            point['p'] = zyx[::-1]
            point['on_patch'] = {'id': patch_id, 'ij': [0.0, float(point_id)]}
            node_ids[id(point)] = len(refs)
            refs.append(zyx)
    patch_points = [
        [coordinates[(10, 0)], coordinates[(10, 1)]],
        [coordinates[(11, 0)], coordinates[(11, 1)]],
    ]
    crossing_map, atlas, patches = _setup(
        refs, patch_points, [[0, 1], [1, 2], [2, 3]])

    cross, _, groups, _, _ = materialize_fiber_fit_inputs(
        {'a': a, 'b': b}, {'p1': None, 'p2': None}, z_begin=-10, z_end=10,
        z_margin=0, min_point_spacing=0, use_adjacent_links=True)
    merged, = cross
    for point in merged['points'].values():
        point['_theta_node_id'] = node_ids[id(point)]
    cfg = _cfg()
    strata = losses.build_pcl_sampling_strata(['fibers'], cfg)

    def loss():
        return losses.get_patch_rel_winding_loss(
            _IdentityTransform(), torch.tensor(DR), patches, atlas, [merged],
            strata, crossing_map=crossing_map, cfg=cfg, z_begin=-1, z_end=1).item()

    # The target across the junction is one winding, matching the geometry.
    assert loss() < 1e-4
    # Treating the link as same-winding (the old behaviour) would demand the
    # V points sit on the H winding, a full winding spacing off.
    for point in b['points'].values():
        point['winding_annotation'] = 0.0
    assert loss() > 0.9 * DR
