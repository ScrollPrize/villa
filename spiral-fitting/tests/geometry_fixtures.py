"""Shared CPU geometry and fit contexts for input revisions and relinking."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from config import Config, FitConfig
from dt_targets import DtTargetCacheManager
from fit_spiral import FitContext, PatchAtlas, _UnattachedPclStripList
from tifxyz import Patch


def _point(point_id, cid, zyx, attached_to=None):
    z, y, x = zyx
    point = {
        'id': point_id, 'collectionId': cid, 'p': [x, y, z],
        'zyx': np.asarray(zyx, dtype=np.float32),
        'winding_annotation': float('nan'),
    }
    if attached_to is not None:
        point['on_patch'] = {'id': attached_to, 'distance': 0.0, 'ij': [0, 0]}
    return point


def _flat_patch(z, y0, x0, size=5, spacing=10.0):
    grid = torch.zeros((size, size, 3), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            grid[i, j] = torch.tensor([z, y0 + i * spacing, x0 + j * spacing])
    return Patch(grid, torch.ones(3), None, None)


def _regular_pcl(cid, zyxs):
        return {
            'id': 3, 'name': 'drawn', 'source_file': '/inputs/drawn.json',
            'sampling_group': '/inputs/drawn.json',
            'metadata': {
                'winding_is_absolute': False, 'input_role': 'legacy',
                'resident_collection_id': cid,
            },
            'points': {i: _point(i, cid, zyx) for i, zyx in enumerate(zyxs)},
        }


def _context(**overrides):
    context = FitContext.__new__(FitContext)
    context.config = Config().as_dict()
    context.config.update({'z_begin': 0, 'z_end': 200})
    context.config.update(overrides)
    context.shell_map = None
    context.shell_envelope = None
    context.shell_outer_winding_idx = None
    context.shell_valid_zyxs_gpu = None
    context.shell_patch = None
    context.tracks = []
    context.prepared_main_tracks = None
    context.verified_patches = {}
    context.verified_patches_list = []
    context.unverified_patches = None
    context.unverified_patches_list = []
    context.unverified_patch_sampling_probabilities = None
    context.unverified_patch_atlas = None
    context.cross_patch_pcls = []
    context.unattached_pcl_strips = _UnattachedPclStripList()
    context.unattached_strip_sampling_groups = []
    context.resolved_links = []
    context.link_components = []
    context.fiber_catalog = {}
    context.regular_pcl_catalog = {}
    context.fiber_direction_samples = None
    context.dt_target_cache_manager = SimpleNamespace(
        update_interval=100, reset=Mock())
    context.theta_crossing_map = SimpleNamespace(invalidate=Mock())
    context._rebuild_pcl_sampling_strata = Mock()
    context._refresh_trusted_geometry = Mock()
    context._build_theta_crossing_map = Mock(return_value=[])
    context._make_shell_polar_map = Mock(return_value='rebuilt shell map')
    return context


@pytest.fixture
def context(monkeypatch):
    import point_collection
    monkeypatch.setattr(point_collection, 'can_use_surface_index_backend', lambda patches: False)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda states: None)
    ctx = FitContext.__new__(FitContext)
    ctx.config = FitConfig(Config({'influence_enabled': False, 'z_begin': 0, 'z_end': 200,
        'patch_erode_patches': 0, 'pcl_unattached_pcl_min_point_spacing': 0}).as_dict())
    ctx.device = torch.device('cpu')
    ctx.progress = None
    ctx.non_liftable_patch_paths = set()
    ctx._source_verified_patches = {'baseline': _flat_patch(50, 10, 10)}
    ctx._source_unverified_patches = {}
    pcl = _regular_pcl(5, [[50, 20, 20], [50, 30, 30]])
    ctx._source_point_collections = {5: pcl}
    ctx.next_id = 6
    ctx.verified_patches = {key: copy.copy(patch)
                            for key, patch in ctx._source_verified_patches.items()}
    ctx.verified_patches_list = list(ctx.verified_patches.values())
    ctx._prepare_patch_sampling_cache(ctx.verified_patches_list)
    ctx.patch_atlas = PatchAtlas(ctx.verified_patches, 'cpu').materialize()
    ctx.unverified_patch_atlas = None
    ctx.dt_target_whole_object = False
    ctx.dt_target_cache_manager = DtTargetCacheManager(100)
    ctx.using_tracks = False
    ctx.slice_to_spiral_transform = lambda x: x
    ctx.optimiser = torch.optim.Adam([torch.nn.Parameter(torch.tensor([1.0]))])
    ctx.influence_state = None
    ctx.run_dt_resume_iteration = None
    ctx.dist = SimpleNamespace(is_main_process=False)
    ctx.interactive_driver = None
    ctx.verified_patches_path = ''
    ctx.unverified_patches_path = ''
    ctx._derive_point_inputs(ctx.verified_patches, copy.deepcopy(ctx._source_point_collections), {})
    for name, value in vars(_context()).items():
        if not name.startswith('_') and not hasattr(ctx, name):
            setattr(ctx, name, value)
    return ctx
