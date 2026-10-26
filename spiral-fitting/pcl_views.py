"""Runtime training views of point collections and fibers.

These helpers turn point collections whose points already carry their patch
attachments (``point['on_patch']``) into the views the losses consume:
cross-patch collections grouped by patch, unattached strips, and fiber-link
components. They do not read files or link points to patches, so every input
source shares them.
"""
import copy

import numpy as np

from point_collection import normalise_pcl_winding_annotations
from spiral_helpers import (
    SequenceChain,
    _decimate_ordered_points_min_spacing,
    build_link_components,
    classify_fiber_hv,
    merge_linked_point_collections,
    point_attachments,
    resolve_fiber_links,
)


class _UnattachedPclStripList(list):
    """List of unattached-pcl strip dicts, with a slot for an attached `.flat`
    GPU bundle that batched satisfaction / winding-range computations reuse."""
    pass


def catalog_copy_of_pcl(pcl):
    """Copy a regular PCL without derived chain and patch-view state."""
    return copy.deepcopy({
        key: value for key, value in pcl.items()
        if key not in ('chain', 'points_by_patch')
    })


def longest_run_in_z_window(sorted_items, z_begin, z_end, z_margin):
    """Return the longest contiguous id-sorted run inside the z window."""
    best_start = best_end = run_start = 0
    for position, (_, point) in enumerate(sorted_items):
        z = point['zyx'][0]
        if z_begin - z_margin <= z < z_end + z_margin:
            if position + 1 - run_start > best_end - best_start:
                best_start, best_end = run_start, position + 1
        else:
            run_start = position + 1
    return sorted_items[best_start:best_end]


def regular_unattached_strip(pcl_id, pcl, min_point_spacing):
    """Materialize the unattached-loss strip of a regular PCL."""
    sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
    if len(sorted_items) < 2:
        return None
    zyxs = np.stack(
        [point['zyx'] for _, point in sorted_items], axis=0).astype(np.float32)
    windings = np.array(
        [point['winding_annotation'] for _, point in sorted_items],
        dtype=np.float32)
    zyxs, keep = _decimate_ordered_points_min_spacing(
        zyxs, min_point_spacing, return_indices=True,
        force_keep={len(zyxs) - 1})
    metadata = pcl.get('metadata', {})
    return {
        'id': pcl_id,
        'name': pcl.get('name'),
        'source_file': pcl.get('source_file'),
        'zyxs': zyxs,
        'windings': windings[keep],
        'link_points': {},
        'logical_input_kind': metadata.get('logical_input_kind'),
        'logical_input_id': metadata.get('logical_input_id'),
        'logical_input_revision': metadata.get('logical_input_revision'),
    }


def group_points_by_patch(pcl, verified_patches):
    """Group a cross-patch PCL's attached points by patch for the winding loss.

    Patches are ordered by the first attached point that hits them when
    scanning the pcl's points in int(key) order; within each patch, points are
    also in int(key) order. Attachments to patches outside ``verified_patches``
    are ignored. A point with several ``attachments`` enters each patch's list
    as an attachment view, ``{'on_patch': ..., 'physical': point}``, so all its
    attachments share the point's chain and theta node.
    """
    points_by_patch = {}
    for _, point in sorted(pcl['points'].items(), key=lambda kv: int(kv[0])):
        for on_patch in point_attachments(point):
            if on_patch['id'] not in verified_patches:
                continue
            entry = (point if 'attachments' not in point
                     else {'on_patch': on_patch, 'physical': point})
            points_by_patch.setdefault(on_patch['id'], []).append(entry)
    pcl['points_by_patch'] = points_by_patch


def regular_pcl_views(point_collections, verified_patches, *, z_begin, z_end,
                      z_margin, min_point_spacing):
    """Classify linked regular PCLs into cross-patch and unattached views.

    Each collection is classified from how its points attach to patches:

    - >= 2 attached points => a cross-patch pcl (winding-number loss), using
      only its attached points (grouped by patch);
    - >= 1 unattached point => an unattached pcl (unattached loss), using the
      entire pcl, trimmed to the longest id-sorted run inside
      [z_begin - z_margin, z_end + z_margin) and dropped with < 2 points left.

    A pcl can fall into both sets. When it does, the unattached entry is an
    independent copy so its z-roi trimming / annotation normalisation cannot
    perturb the cross-patch entry's points_by_patch (which is built from all
    attached points, regardless of z).

    Exception: pcls flagged metadata.winding_is_absolute carry absolute winding
    annotations and are always consumed as cross-patch pcls (never unattached),
    retained even when they hold a single point. Points of theirs that failed
    to attach to a patch only produce a warning -- they carry no winding target
    and never enter points_by_patch -- and every *attached* point must carry an
    explicit, positive winding annotation (an absolute pcl must not fall back
    to winding 0).

    The containers in ``point_collections`` become the cross-patch views.
    Returns ``(cross_patch, strips, strip_sampling_groups)``: cross-patch pcls
    keyed by collection id, and the unattached strips with their parallel
    sampling groups.
    """
    cross_patch = {}
    unattached = {}
    for pid, pcl in point_collections.items():
        num_attached = sum(1 for point in pcl['points'].values() if 'on_patch' in point)
        num_unattached = len(pcl['points']) - num_attached
        if pcl.get('metadata', {}).get('winding_is_absolute', False):
            if num_unattached > 0:
                print(
                    f'WARNING: winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has '
                    f'{num_unattached} of {len(pcl["points"])} points not attached to any patch; '
                    f'dropping the unattached points'
                )
            # Validate only the attached points -- unattached ones are dropped above and never
            # enter points_by_patch, so their annotations are irrelevant.
            attached_points = [point for point in pcl['points'].values() if 'on_patch' in point]
            num_unannotated = sum(1 for point in attached_points if not np.isfinite(point['winding_annotation']))
            if num_unannotated:
                raise ValueError(
                    f'winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has {num_unannotated} of '
                    f'{len(attached_points)} attached points without a winding annotation; absolute pcls '
                    f'must give every winding number explicitly'
                )
            num_non_positive = sum(1 for point in attached_points if point['winding_annotation'] <= 0)
            if num_non_positive:
                raise ValueError(
                    f'winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has {num_non_positive} of '
                    f'{len(attached_points)} attached points with a non-positive winding annotation; '
                    f'absolute winding numbers must be > 0'
                )
            cross_patch[pid] = pcl
            continue
        if num_attached >= 2:
            cross_patch[pid] = pcl
        if num_unattached >= 1:
            unattached[pid] = copy.deepcopy(pcl) if num_attached >= 2 else pcl

    dropped = 0
    for pid in list(unattached):
        pcl = unattached[pid]
        kept_items = longest_run_in_z_window(
            sorted(pcl['points'].items(), key=lambda kv: int(kv[0])),
            z_begin, z_end, z_margin)
        if len(kept_items) < 2:
            del unattached[pid]
            dropped += 1
        else:
            pcl['points'] = dict(kept_items)
    if dropped:
        print(f'dropped {dropped} unattached pcls with <2 points in z-roi')

    normalise_pcl_winding_annotations(cross_patch)
    normalise_pcl_winding_annotations(unattached)

    for pcl in cross_patch.values():
        group_points_by_patch(pcl, verified_patches)

    # Strips are id-sorted; with min_point_spacing > 0 they are decimated
    # greedily so consecutive kept points are at least that far apart. The
    # first and last points are always kept.
    strips = _UnattachedPclStripList()
    strip_sampling_groups = []
    for pcl_id, pcl in unattached.items():
        strip = regular_unattached_strip(pcl_id, pcl, min_point_spacing)
        if strip is None:
            continue
        strips.append(strip)
        strip_sampling_groups.append(pcl['sampling_group'])
    return cross_patch, strips, strip_sampling_groups


def prepare_fiber_catalog(fiber_catalog):
    """Stamp fiber identity, chains and ZYX on the resident fiber catalog.

    ``fiber_catalog`` is ordered by logical input id and owns the canonical
    fiber containers. Returns them keyed by resident collection id.
    """
    point_collections = {}
    for logical_id, pcl in fiber_catalog.items():
        logical_id = str(logical_id)
        metadata = pcl.setdefault('metadata', {})
        metadata['logical_input_kind'] = 'fiber'
        metadata['logical_input_id'] = logical_id
        metadata['winding_is_absolute'] = False
        metadata.setdefault('input_role', 'fiber')
        pcl.setdefault('file_basename', f'{logical_id}.json')
        pcl['sampling_group'] = 'fibers'
        pcl['chain'] = SequenceChain(pcl)
        for point in pcl['points'].values():
            point['collectionId'] = pcl['id']
            if 'zyx' not in point:
                point['zyx'] = np.asarray(
                    [point['p'][2], point['p'][1], point['p'][0]],
                    dtype=np.float32)
        if pcl['id'] in point_collections:
            raise ValueError(f'duplicate resident fiber collection id {pcl["id"]!r}')
        point_collections[pcl['id']] = pcl
    return point_collections


def fiber_strip(cid, pcl, link_ids, *, z_begin, z_end, z_margin,
                min_point_spacing, vertical_min_z_fraction,
                vertical_min_auto_certainty, vertical_radial_offset):
    """Materialize one fiber's unattached strip, or None with < 2 points.

    The strip is the longest id-sorted run inside the z window, decimated to
    ``min_point_spacing`` while keeping the last point and every point whose
    id is in ``link_ids``; ``link_points`` maps those kept ids to strip rows.
    """
    kept_items = longest_run_in_z_window(
        sorted(pcl['points'].items(), key=lambda item: int(item[0])),
        z_begin, z_end, z_margin)
    if len(kept_items) < 2:
        return None

    force_keep = {
        position for position, (point_id, _) in enumerate(kept_items)
        if int(point_id) in link_ids
    }
    zyxs = np.stack(
        [point['zyx'] for _, point in kept_items], axis=0).astype(np.float32)
    windings = np.asarray(
        [point['winding_annotation'] for _, point in kept_items],
        dtype=np.float32)
    bake_scale = np.asarray(
        [point.get('radial_offset_bake_scale', 1.0)
         for _, point in kept_items],
        dtype=np.float32)
    zyxs, keep = _decimate_ordered_points_min_spacing(
        zyxs, min_point_spacing, return_indices=True,
        force_keep=force_keep | {len(zyxs) - 1})
    windings = windings[keep]
    bake_scale = bake_scale[keep]
    link_points = {
        int(kept_items[original_position][0]): strip_position
        for strip_position, original_position in enumerate(keep)
        if original_position in force_keep
    }
    metadata = pcl.get('metadata', {})
    hv_tag = classify_fiber_hv(
        metadata.get('hv_classification'), zyxs,
        min_z_fraction=vertical_min_z_fraction,
        min_auto_certainty=vertical_min_auto_certainty)
    return {
        'id': cid,
        'name': pcl.get('name'),
        'source_file': pcl.get('source_file'),
        'zyxs': zyxs,
        'windings': windings,
        'radial_offsets': np.full(
            len(zyxs),
            float(vertical_radial_offset) if hv_tag == 'V' else 0.0,
            dtype=np.float32),
        'radial_offset_bake_scale': bake_scale,
        'link_points': link_points,
        'hv_tag': hv_tag,
        'is_vertical': hv_tag == 'V',
        'logical_input_kind': 'fiber',
        'logical_input_id': metadata.get('logical_input_id'),
        'logical_input_revision': metadata.get('logical_input_revision'),
    }


def fiber_training_views(point_collections, resolved_links, verified_patches,
                         *, z_begin, z_end, z_margin, min_point_spacing,
                         vertical_min_z_fraction=0.8,
                         vertical_min_auto_certainty=0.5,
                         vertical_radial_offset=0.0, strip_every_fiber=False):
    """Derive the fiber training views from prepared fibers and resolved links.

    ``point_collections`` are the prepared catalog fibers (prepare_fiber_catalog)
    in catalog order; ``resolved_links`` are same-winding links
    ``{'a_coll', 'a_point', 'b_coll', 'b_point', 'pending'}`` between them.
    A fiber with two or more patch attachments is a cross-patch view (linked
    fibers merge into one component). Strips come from fibers with an
    unattached point or a link, or from every fiber with
    ``strip_every_fiber``; each needs two points in the z window.
    Returns ``(cross_patch, strips, sampling_groups, link_components)``.
    """
    link_components = build_link_components(resolved_links)
    catalog_position = {
        cid: position for position, cid in enumerate(point_collections)
    }
    link_components = [
        (sorted(member_cids, key=catalog_position.__getitem__), member_links)
        for member_cids, member_links in link_components
    ]
    link_components.sort(
        key=lambda component: min(
            catalog_position[cid] for cid in component[0]))
    linked_cids = {
        cid for member_cids, _ in link_components for cid in member_cids
    }

    cross_patch = {}
    strip_sources = []
    for cid, pcl in point_collections.items():
        num_attached = sum(
            len(point_attachments(point)) for point in pcl['points'].values())
        num_unattached = sum(
            1 for point in pcl['points'].values() if not point_attachments(point))
        if num_attached >= 2:
            cross_patch[cid] = pcl
        if strip_every_fiber or num_unattached >= 1 or cid in linked_cids:
            strip_sources.append((cid, pcl))

    if link_components:
        cross_patch, _ = merge_linked_point_collections(
            point_collections, link_components, cross_patch)

    # Fibers carry no winding annotations. Their point dictionaries are
    # shared by the catalog and derived PCLs, so normalize them directly after
    # link resolution; subsequent materializations explicitly resolve catalog
    # fibers as unannotated.
    for pcl in point_collections.values():
        pcl['has_winding_annotations'] = False
        for point in pcl['points'].values():
            point['winding_annotation'] = 0.0

    for pcl in cross_patch.values():
        group_points_by_patch(pcl, verified_patches)

    link_point_ids_by_coll = {}
    for link in resolved_links:
        link_point_ids_by_coll.setdefault(link['a_coll'], set()).add(
            link['a_point'])
        link_point_ids_by_coll.setdefault(link['b_coll'], set()).add(
            link['b_point'])

    strips = _UnattachedPclStripList()
    sampling_groups = []
    for cid, pcl in strip_sources:
        strip = fiber_strip(
            cid, pcl, link_point_ids_by_coll.get(cid, ()),
            z_begin=z_begin, z_end=z_end, z_margin=z_margin,
            min_point_spacing=min_point_spacing,
            vertical_min_z_fraction=vertical_min_z_fraction,
            vertical_min_auto_certainty=vertical_min_auto_certainty,
            vertical_radial_offset=vertical_radial_offset)
        if strip is None:
            continue
        strips.append(strip)
        sampling_groups.append(pcl['sampling_group'])

    return list(cross_patch.values()), strips, sampling_groups, link_components


def materialize_fiber_fit_inputs(
        fiber_catalog, verified_patches, *, z_begin, z_end, z_margin,
        min_point_spacing, use_links=True, use_pending_links=False,
        vertical_min_z_fraction=0.8, vertical_min_auto_certainty=0.5,
        vertical_radial_offset=0.0):
    """Derive every CPU training view from the resident fiber catalog.

    ``fiber_catalog`` is ordered by logical input id and owns the canonical
    fiber containers. The returned cross-patch PCLs and strips refer to their
    geometry, but trimming and decimation never mutate the catalog's point
    membership. Re-running this function over the same catalog therefore
    produces the same graph and ordered training views.

    Each strip carries ``hv_tag`` ('V', 'H', or None; see
    spiral_helpers.classify_fiber_hv) and ``is_vertical`` for radial-offset updates.
    ``radial_offsets`` holds the
    per-point radial target offset in scroll voxels along the sheet normal,
    positive in the increasing-winding direction of the fitted spiral (the
    scan-space winding gradient, not the line to the umbilicus):
    ``vertical_radial_offset`` on vertical strips (their back-face position
    relative to the fitted sheet, away from the umbilicus), zero everywhere
    else.
    ``radial_offset_bake_scale`` carries each point's accumulated normal
    stretch through the frozen constraint-bake stack (1 before any bake),
    read from the catalog point dicts (see accumulate_radial_offset_bake_scale)
    so the physical offset can be expressed in the resident frame.
    """
    point_collections = prepare_fiber_catalog(fiber_catalog)
    resolved_links = []
    if use_links:
        resolved_links = resolve_fiber_links(
            point_collections, include_pending=use_pending_links,
            assume_unannotated=True)
    cross_patch, strips, sampling_groups, link_components = \
        fiber_training_views(
            point_collections, resolved_links, verified_patches,
            z_begin=z_begin, z_end=z_end, z_margin=z_margin,
            min_point_spacing=min_point_spacing,
            vertical_min_z_fraction=vertical_min_z_fraction,
            vertical_min_auto_certainty=vertical_min_auto_certainty,
            vertical_radial_offset=vertical_radial_offset)
    return (cross_patch, strips, sampling_groups,
            resolved_links, link_components)
