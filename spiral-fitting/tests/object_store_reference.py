"""Synthetic fit inputs and canonical summaries of the constraints they produce.

``FIXTURE`` describes a small scroll: four patches on an Archimedean spiral,
absolute/relative/same-winding point collections, and three fibers, two of
them joined by a same-winding branch. Every attached point lies on exactly one
patch vertex and every other point lies half a winding from any sheet, so
attachments do not depend on linker tie-breaks.

``constraint_summary`` and ``loss_summary`` reduce a loaded FitContext to
JSON-comparable values. tests/data/object_store_reference.json holds them as
produced by the file loader and its proximity linker; the object-store adapter
must reproduce them from identical attachments.

    python tests/object_store_reference.py   # rewrite the reference file
"""
import json
import math
import os
import sys

import numpy as np
import torch

SPIRAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPIRAL_DIR)

REFERENCE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data',
    'object_store_reference.json')

DR = 12.0
CENTRE_YX = (150.0, 150.0)
BASE_SHAPE_ZYX = (64, 300, 300)
GRID_STEP = 4.0
SEED = 1234


def spiral_zyx(theta, winding, z):
    """Volume ZYX of the spiral sheet ``winding`` at ``theta`` and ``z``."""
    radius = DR * (winding + theta / (2 * math.pi))
    return [float(z), CENTRE_YX[0] + math.sin(theta) * radius,
            CENTRE_YX[1] + math.cos(theta) * radius]


def _patch(name, uuid, winding, theta_begin, theta_end):
    zs = np.arange(4.0, 60.0 + 1e-6, GRID_STEP)
    radius = DR * (winding + (theta_begin + theta_end) / (4 * math.pi))
    thetas = np.arange(theta_begin, theta_end + 1e-9, GRID_STEP / radius)
    grid = np.asarray([[spiral_zyx(theta, winding, z) for theta in thetas]
                       for z in zs], dtype=np.float32)
    return {'name': name, 'uuid': uuid, 'winding': winding,
            'thetas': thetas, 'zs': zs, 'zyxs': grid,
            'scale': [1 / GRID_STEP, 1 / GRID_STEP]}


def _vertex(patch, row, col):
    """A point exactly on a patch vertex, and that vertex's (row, col)."""
    return (spiral_zyx(patch['thetas'][col], patch['winding'],
                       patch['zs'][row]), (row, col))


PATCHES = [
    _patch('patch_a', '00000000-0000-4000-8000-0000000000a1', 8, 0.3, 1.3),
    _patch('patch_b', '00000000-0000-4000-8000-0000000000a2', 8, 1.1, 2.1),
    _patch('patch_c', '00000000-0000-4000-8000-0000000000a3', 9, 0.5, 1.5),
    _patch('patch_d', '00000000-0000-4000-8000-0000000000a4', 10, 2.0, 3.0),
]
_A, _B, _C, _D = PATCHES


def _between(theta, winding, z):
    return spiral_zyx(theta, winding + 0.5, z)


def _pcl_points(entries):
    """{point id: {'zyx', 'wind', 'on'}}; ``on`` names the patch it lies on."""
    return {point_id: dict(zip(('zyx', 'wind', 'on'), entry))
            for point_id, entry in entries}


PCL_DOCUMENTS = [
    {'stem': 'abs_winding', 'role': 'absolute', 'collections': [{
        'name': 'absolute', 'uuid': '00000000-0000-4000-8000-0000000000b1',
        'points': _pcl_points([
            (2, (_vertex(_A, 5, 6)[0], 8.0, 'patch_a')),
            (5, (_vertex(_C, 6, 8)[0], 9.0, 'patch_c')),
            (9, (_vertex(_D, 4, 10)[0], 10.0, 'patch_d')),
        ])}]},
    {'stem': 'relative_windings', 'role': 'relative', 'collections': [{
        'name': 'relative', 'uuid': '00000000-0000-4000-8000-0000000000b2',
        'points': _pcl_points([
            (0, (_vertex(_A, 8, 4)[0], 0.0, 'patch_a')),
            (3, (_between(0.75, 8, 36.0), 0.5, None)),
            (4, (_between(0.8, 8, 38.0), 0.5, None)),
            (7, (_vertex(_C, 9, 4)[0], 1.0, 'patch_c')),
        ])}]},
    {'stem': 'same_windings', 'role': 'same_winding', 'collections': [
        {'name': 'same-a', 'uuid': '00000000-0000-4000-8000-0000000000b3',
         'points': _pcl_points([
             (1, (_vertex(_A, 3, 20)[0], None, 'patch_a')),
             (4, (_vertex(_B, 3, 18)[0], None, 'patch_b')),
             (6, (spiral_zyx(2.5, 8, 16.0), None, None)),
             (8, (spiral_zyx(2.55, 8, 16.0), None, None)),
         ])},
        # Wholly unattached, sampled more densely than the strip spacing.
        {'name': 'same-free', 'uuid': '00000000-0000-4000-8000-0000000000b4',
         'points': _pcl_points([
             (10 + k, (spiral_zyx(2.2 + 0.02 * k, 9, 40.0), None, None))
             for k in range(12)
         ])},
    ]},
]


def _fiber_line(samples):
    """Line ZYX from (theta, winding, z) samples."""
    return [spiral_zyx(*sample) for sample in samples]


def _f1_samples():
    thetas = np.round(np.arange(0.4, 1.95 + 1e-9, 0.05), 6)
    samples = []
    for theta in thetas:
        on_sheet = abs(theta - 0.6) < 1e-6 or abs(theta - 1.8) < 1e-6
        samples.append((float(theta), 8 if on_sheet else 8.5, 30.0))
    return samples


def _f2_samples():
    return [(1.95, 8 if z == 46.0 else 8.5, z)
            for z in np.arange(14.0, 58.0 + 1e-6, 4.0)]


_F1 = _fiber_line(_f1_samples())
_F2 = _fiber_line(_f2_samples())
_F3 = _fiber_line([(float(t), 9.5, 10.0)
                   for t in np.arange(0.6, 1.4 + 1e-9, 0.05)])

FIBERS = [
    # Horizontal, touching patch A at theta 0.6 and patch B at 1.8.
    {'name': 'fiber_1', 'uuid': '00000000-0000-4000-8000-0000000000c1',
     'line': _F1, 'controls': [0, 15, len(_F1) - 1],
     'contacts': {4: 'patch_a', 28: 'patch_b'}},
    # Vertical, touching patch B at z 46; joined to fiber_1 where they meet.
    {'name': 'fiber_2', 'uuid': '00000000-0000-4000-8000-0000000000c2',
     'line': _F2, 'controls': [0, 4, len(_F2) - 1],
     'contacts': {8: 'patch_b'}},
    # Unlinked and unattached.
    {'name': 'fiber_3', 'uuid': '00000000-0000-4000-8000-0000000000c3',
     'line': _F3, 'controls': [0, len(_F3) - 1], 'contacts': {}},
]
# (fiber, control position, fiber, control position), same winding.
FIBER_BRANCHES = [('fiber_1', 2, 'fiber_2', 1)]

CONFIG_OVERRIDES = {
    'z_begin': 0,
    'z_end': 64,
    'optimizer_random_seed': SEED,
    'pcl_fiber_min_point_spacing': 0.0,
    'pcl_unattached_pcl_min_point_spacing': 3.0,
    'input_use_normals': False,
    'input_use_gradient_magnitude': False,
    'input_use_winding_inference': False,
    'input_use_outer_shell': False,
    'input_use_tracks': False,
    'input_use_fiber_directions': False,
    'loss_weight_fiber_directions': 0.0,
    'sample_count_relative_winding_pcls': 8,
    'sample_count_absolute_winding_pcls': 2,
    'sample_count_unattached_pcls_per_step': 8,
    'sample_count_points_per_patch': 16,
}


def fit_config():
    from config import Config
    config = Config().as_dict()
    config.update(CONFIG_OVERRIDES)
    return config


def scroll_spec():
    from fit_session import ScrollSpec
    return ScrollSpec(name='synthetic', voxel_size_um=1.0,
                      spiral_outward_sense='CW',
                      base_shape_zyx=BASE_SHAPE_ZYX)


def write_umbilicus(path):
    with open(path, 'w') as stream:
        json.dump({'control_points': [
            {'z': z, 'y': CENTRE_YX[0], 'x': CENTRE_YX[1]} for z in (0, 64)]},
            stream)


def write_file_dataset(root):
    """Write FIXTURE as a conventional file dataset; return SpiralInputPaths."""
    from PIL import Image
    from fit_session import PclInputSpec, PclRole, SpiralInputPaths

    patches_root = os.path.join(root, 'verified_patches')
    for patch in PATCHES:
        path = os.path.join(patches_root, patch['name'])
        os.makedirs(path)
        for axis, name in enumerate('zyx'):
            Image.fromarray(np.ascontiguousarray(
                patch['zyxs'][..., axis])).save(os.path.join(path, f'{name}.tif'))
        with open(os.path.join(path, 'meta.json'), 'w') as stream:
            json.dump({'uuid': patch['name'], 'scale': patch['scale']}, stream)

    roles = {'absolute': PclRole.ABSOLUTE, 'relative': PclRole.RELATIVE,
             'same_winding': PclRole.SAME_WINDING}
    pcl_specs = []
    for document in PCL_DOCUMENTS:
        collections = {}
        for index, collection in enumerate(document['collections']):
            collections[str(index)] = {'name': collection['name'], 'points': {
                str(point_id): {'p': point['zyx'][::-1],
                                **({} if point['wind'] is None
                                   else {'wind_a': point['wind']})}
                for point_id, point in collection['points'].items()}}
        path = os.path.join(root, f'{document["stem"]}.json')
        with open(path, 'w') as stream:
            json.dump({'vc_pointcollections_json_version': '1',
                       'collections': collections}, stream)
        pcl_specs.append(PclInputSpec(path, roles[document['role']]))

    fibers_root = os.path.join(root, 'fibers')
    os.makedirs(fibers_root)
    branches = {fiber['name']: [] for fiber in FIBERS}
    for a, a_control, b, b_control in FIBER_BRANCHES:
        branches[a].append({'control_point_index': a_control,
                            'branch_file': f'{b}.json',
                            'branch_control_point_index': b_control})
        branches[b].append({'control_point_index': b_control,
                            'branch_file': f'{a}.json',
                            'branch_control_point_index': a_control})
    for fiber in FIBERS:
        line_xyz = [zyx[::-1] for zyx in fiber['line']]
        with open(os.path.join(fibers_root, f'{fiber["name"]}.json'), 'w') as stream:
            json.dump({
                'type': 'vc3d_fiber', 'version': 1, 'name': fiber['name'],
                'coordinate_base_shape_zyx': list(BASE_SHAPE_ZYX),
                'control_points': [line_xyz[i] for i in fiber['controls']],
                'line_points': line_xyz,
                'branches': branches[fiber['name']],
            }, stream)

    umbilicus = os.path.join(root, 'umbilicus.json')
    write_umbilicus(umbilicus)
    return SpiralInputPaths(
        dataset_root=root, umbilicus=umbilicus, pcls=tuple(pcl_specs),
        fibers=fibers_root, verified_patches=patches_root)


def file_context(root):
    """A FitContext whose host inputs came from the file loader."""
    from config import FitConfig
    from fit_spiral import FitContext
    paths = write_file_dataset(root)
    context = FitContext(FitConfig(fit_config()), scroll=scroll_spec(),
                         paths=paths, cache_dir=os.path.join(root, 'cache'))
    context.load_host_inputs()
    return context


def _r(values, digits=4):
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def _group(group):
    return os.path.splitext(os.path.basename(str(group)))[0]


def _point_key(point):
    """Name-independent identity of a physical point: its rounded ZYX."""
    return _r(point['zyx'], 3)


def constraint_summary(context, patch_names=None):
    """JSON-comparable runtime constraints; ``patch_names`` maps runtime
    patch ids to fixture names (identity when None)."""
    name_of = (lambda pid: pid) if patch_names is None else patch_names.__getitem__
    patches = []
    for pid, patch in context.verified_patches.items():
        valid = patch.valid_vertex_mask.numpy()
        patches.append({
            'name': name_of(pid),
            'shape': list(patch.zyxs.shape),
            'valid_vertices': int(valid.sum()),
            'valid_sum': _r(patch.zyxs.numpy()[valid].sum(axis=0), 2),
            'scale': _r(patch.scale.numpy()),
            'area': round(float(patch.area), 3),
            'sampling_area': round(float(patch._sampling_area), 3),
        })
    cross_patch = []
    for pcl in context.cross_patch_pcls:
        cross_patch.append({
            'group': _group(pcl['sampling_group']),
            'absolute': bool(pcl.get('metadata', {}).get('winding_is_absolute')),
            'members': len(pcl.get('link_member_cids', ())),
            'points': [[_point_key(p), round(float(p['winding_annotation']), 4)]
                       for _, p in sorted(pcl['points'].items(),
                                          key=lambda item: int(item[0]))],
            'points_by_patch': [
                [name_of(pid), [[_point_key(p.get('physical', p)),
                                 _r(p['on_patch']['ij'], 3)] for p in points]]
                for pid, points in pcl['points_by_patch'].items()],
            'chain': [_point_key(p) for p in pcl['chain'].iter_chain()],
        })
    strips = []
    for strip, group in zip(context.unattached_pcl_strips,
                            context.unattached_strip_sampling_groups):
        zyxs = np.asarray(strip['zyxs'])
        strips.append({
            'group': _group(group),
            'name': strip.get('name'),
            'zyxs': _r(zyxs, 3),
            'windings': _r(strip['windings']),
            'link_points': sorted(
                _r(zyxs[position], 3)
                for position in strip.get('link_points', {}).values()),
            'hv_tag': strip.get('hv_tag'),
            'radial_offsets': _r(strip.get('radial_offsets', np.zeros(len(zyxs)))),
        })
    return {
        'patches': patches,
        'cross_patch': cross_patch,
        'strips': strips,
        'unattached_components': [list(c) for c in context.unattached_components],
        'unattached_component_groups': [
            _group(g) for g in context.unattached_component_groups],
        'unattached_component_edges': [
            [list(map(int, edge)) for edge in edges]
            for edges in context.unattached_component_edges],
        'cross_patch_strata': [
            [_group(g), s.tolist()] for g, s in zip(
                context.pcl_sampling_strata['cross_patch']['groups'],
                context.pcl_sampling_strata['cross_patch']['strata'])],
        'unattached_strata': [
            [_group(g), s.tolist()] for g, s in zip(
                context.pcl_sampling_strata['unattached']['groups'],
                context.pcl_sampling_strata['unattached']['strata'])],
    }


class _CentredTransform:
    """Slice-to-spiral transform of the fixture: translate the axis to 0."""

    @staticmethod
    def _centre(zyxs):
        return torch.as_tensor(
            [0.0, *CENTRE_YX], dtype=zyxs.dtype, device=zyxs.device)

    def __call__(self, zyxs):
        return zyxs - self._centre(zyxs)

    def inv(self, spiral_zyxs):
        return spiral_zyxs + self._centre(spiral_zyxs)


def loss_summary(context, steps=4):
    """Winding and strip losses over ``steps`` fixed-seed draws."""
    import losses
    from fit_spiral import get_or_build_unattached_pcl_flat

    device = torch.device('cpu')
    transform = _CentredTransform()
    dr = torch.tensor(DR)
    context.device = device
    context.patch_atlas.materialize(device)
    context.unattached_pcl_strips.flat = None
    crossing_map = context._make_theta_crossing_map()
    crossing_map.force_refresh(transform)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    result = {'rel_winding': [], 'abs_winding': [], 'strip_radius': [],
              'strip_dt': []}
    cfg = context.config
    for _ in range(steps):
        result['rel_winding'].append(float(losses.get_patch_rel_winding_loss(
            transform, dr, context.verified_patches, context.patch_atlas,
            context.cross_patch_pcls, context.pcl_sampling_strata['cross_patch'],
            crossing_map=crossing_map, cfg=cfg,
            z_begin=context.z_begin, z_end=context.z_end)))
        result['abs_winding'].append(float(losses.get_patch_abs_winding_loss(
            transform, dr, context.verified_patches, context.patch_atlas,
            context.cross_patch_pcls, crossing_map=crossing_map, cfg=cfg,
            z_begin=context.z_begin, z_end=context.z_end)))
        radius, dt = losses.get_unattached_pcl_strip_losses(
            transform, dr, context.unattached_pcl_strips,
            context.unattached_components, context.unattached_component_edges,
            context.pcl_sampling_strata['unattached'],
            get_or_build_unattached_pcl_flat,
            cfg['sample_count_unattached_pcls_per_step'],
            cfg['sample_count_unattached_pcl_points_per_step'],
            compute_dt=True, crossing_map=crossing_map, cfg=cfg)
        result['strip_radius'].append(float(radius))
        result['strip_dt'].append(float(dt))
    return {key: [round(value, 5) for value in values]
            for key, values in result.items()}


def reference_summary(context, patch_names=None):
    return {'constraints': constraint_summary(context, patch_names),
            'losses': loss_summary(context)}


def load_reference():
    with open(REFERENCE_PATH) as stream:
        return json.load(stream)


if __name__ == '__main__':
    import tempfile
    os.environ.setdefault('FIT_SPIRAL_DEVICE', 'cpu')
    with tempfile.TemporaryDirectory() as root:
        summary = reference_summary(file_context(root))
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    with open(REFERENCE_PATH, 'w') as stream:
        json.dump(summary, stream, indent=1, sort_keys=True)
        stream.write('\n')
    print(f'wrote {REFERENCE_PATH}')
