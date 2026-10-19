"""Trainer policy adapter for vc_reader's general geometry operations.

Sampling/state decisions remain in workers. Native CPU plane intersections
replace their Python plane loops. Final neighbor clearance, frame transforms
and voxelization run on CUDA; compiled losses consume sorted sparse keys.
"""
from collections import OrderedDict
import os
import time

import numpy as np
import torch

from vc_reader.geometry import PolylineSet, VoxelSet, transform_points

ORIGINAL_PLANE = None
ORIGINAL_BANK = None
ORIGINAL_HITS = None


def plane_targets(p, s, t, t_end, pos, frame, planes):
    from vesuvius.neural_tracing.tracers.shared.geometry import interp_at
    stop = min(t_end, t + 2.5 * float(planes[-1]))
    ab = np.zeros((len(planes), 2), np.float32)
    mask = np.zeros(len(planes), np.float32)
    if stop > t:
        # Center in float64 before float32 conversion: world coordinates and
        # cumulative arclengths can be much larger than a training crop.
        arc = np.r_[t, s[(s > t) & (s < stop)], stop]
        local = np.asarray((interp_at(p, s, arc)-pos) @ frame, np.float32)
        curves = PolylineSet(local, [0, len(local)], arclengths=(arc-t).astype(np.float32))
        equations = np.zeros((len(planes), 4), np.float32)
        equations[:, 2], equations[:, 3] = 1., planes
        result = curves.intersect_planes(np.zeros(len(planes), np.int64), equations)
        valid = result['valid']
        ab[valid], mask[valid] = result['points'][valid, :2], 1.
    if os.environ.get('FIBER_BENCH_GEOMETRY_VALIDATE') == '1':
        ref_ab, ref_mask = ORIGINAL_PLANE(p, s, t, t_end, pos, frame, planes)
        np.testing.assert_array_equal(mask, ref_mask)
        np.testing.assert_allclose(ab, ref_ab, atol=5e-4, rtol=2e-5)
    return ab, mask


def bank_targets(self, items, decision_mask=None):
    # Sampling feedback is needed before the next batch can be planned.
    if decision_mask is not None:
        return ORIGINAL_BANK(self, items, decision_mask)
    feedback = ORIGINAL_BANK(self, items, np.zeros(len(items), bool))
    requests = []
    crop, bank = self.cfg.fine, self.neighbors
    lateral, forward = crop.lateral_coords[[0, -1]], crop.forward_coords[[0, -1]]
    corners = np.array([[a, b, z] for a in lateral for b in lateral for z in forward])
    for item in items:
        request = None
        if bank is not None and len(item.get('identity_curve', ())) >= 3:
            fi = item['fiber_ref'][0]
            parent = bank.fibers[fi].points
            origin = parent[0]
            pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
            world_corners = corners @ frame.T+pos
            records = bank.spatial_records(fi, world_corners)
            lines = [r['samples'] for r in records]
            if lines:
                world = np.concatenate(lines)
                # Catalog blocks can contain long annotation tails outside the
                # crop. Reject those before upload using a conservative AABB.
                lo, hi = world_corners.min(0)-1e-7, world_corners.max(0)+1e-7
                world = world[np.all((world >= lo) & (world <= hi), axis=1)]
                # Rebasing before conversion preserves subvoxel precision.
                request = dict(key=(str(bank.root), int(fi)),
                    parent=np.asarray(parent-origin, np.float32),
                    # Float64 is required only through voxel rounding: float32
                    # can turn values just below a half-cell into exact ties.
                    points=np.ascontiguousarray(world-pos, dtype=np.float64),
                    position=np.asarray(pos-origin, np.float32), frame=frame,
                    identity=np.asarray(item['identity_curve'], np.float32),
                    gap=float(np.linalg.norm(np.diff(parent, axis=0), axis=1).max()/2),
                    exclusion=max(bank.exclusion, self.sampling.rule.own_radius))
                if os.environ.get('FIBER_BENCH_GEOMETRY_VALIDATE') == '1':
                    request['reference'] = dict(world=world, parent=parent, pos=pos,
                                                frame=frame, identity=item['identity_curve'])
        requests.append(request)
    feedback['_vc_geometry'] = dict(requests=requests, crop=crop)
    if os.environ.get('FIBER_BENCH_GEOMETRY_VALIDATE') == '1':
        # Avoid advancing the sampling memory a second time during validation.
        shadow = [dict(item, _sampling_feedback_prepared=True) for item in items]
        feedback['_vc_geometry']['expected'] = ORIGINAL_BANK(self, shadow)
    return feedback


def foreign_hits(points, batch, cfg):
    foreign = batch['foreign']
    if foreign.ndim != 2:
        return ORIGINAL_HITS(points, batch, cfg)
    from vesuvius.neural_tracing.tracers.models.follower import feature_grid
    shape = (cfg.fine.depth, cfg.fine.width, cfg.fine.width)
    grid = feature_grid(points.detach().float(), cfg.fine, shape)
    # Reproduce grid_sample's align_corners unnormalization before rounding.
    xyz = torch.stack([((grid[..., a]+1)/2)*(size-1)
                       for a, size in enumerate(reversed(shape))], -1).round()
    valid = torch.isfinite(xyz).all(-1)
    for a, size in enumerate(reversed(shape)):
        valid = valid & (xyz[..., a] >= 0) & (xyz[..., a] < size)
    xyz = torch.where(valid[..., None], xyz, 0).long()
    batch_ids = torch.arange(len(points), device=points.device)[:, None]
    keys = ((batch_ids*shape[0]+xyz[..., 2])*shape[1]+xyz[..., 1])*shape[2]+xyz[..., 0]
    stored = foreign.flatten()
    indices = torch.searchsorted(stored, keys.flatten()).clamp_max(stored.numel()-1)
    return valid & (stored[indices].reshape(keys.shape) == keys)


def install_worker():
    global ORIGINAL_PLANE, ORIGINAL_BANK
    from vesuvius.neural_tracing.tracers.fibers.data import targets, observations
    if ORIGINAL_PLANE is None:
        ORIGINAL_PLANE = targets.plane_targets
        targets.plane_targets = plane_targets
    if ORIGINAL_BANK is None:
        ORIGINAL_BANK = observations.IdentityObservationBuilder.bank_targets
        observations.IdentityObservationBuilder.bank_targets = bank_targets


def install(*, validate=False, placement='prefetch'):
    global ORIGINAL_HITS
    os.environ['FIBER_BENCH_GEOMETRY'] = '1'
    os.environ['FIBER_BENCH_GEOMETRY_VALIDATE'] = '1' if validate else '0'
    install_worker()
    from vesuvius.neural_tracing.tracers.models import targets
    if ORIGINAL_HITS is None:
        ORIGINAL_HITS = targets.foreign_hits
        targets.foreign_hits = foreign_hits
    geometry = CudaGeometry()
    geometry.placement = placement
    if placement == 'prefetch':
        from vesuvius.neural_tracing.tracers.fibers.train import train
        from vc_reader import _native
        from fiber_timeline import span
        original = train.DecisionBatchPrefetch.collect
        # Reuse the trainer's existing one-batch producer. Its completed CUDA
        # outputs are consumed on the model stream, with allocator ownership
        # recorded by the benchmark before model work is submitted.
        stream = torch.cuda.Stream()

        def collect(prefetch):
            batches = original(prefetch)
            with torch.cuda.stream(stream):
                _native._set_stream(stream.cuda_stream)
                try:
                    for batch in batches:
                        began = time.perf_counter()
                        with span('geometry.prepare'):
                            geometry.materialize(batch)
                        batch['_bench_geometry_timing'] = dict(began=began, ready=time.perf_counter())
                finally:
                    stream.synchronize()
                    _native._set_stream(0)
            return batches
        train.DecisionBatchPrefetch.collect = collect
    return geometry


def nearest_outputs(points):
    n = len(points)
    result = dict(points=torch.empty_like(points),
        distances=torch.empty(n, device=points.device), arclengths=torch.empty(n, device=points.device),
        indices=torch.empty(n, dtype=torch.int64, device=points.device),
        curve_ids=torch.empty(n, dtype=torch.int64, device=points.device),
        valid=torch.empty(n, dtype=torch.bool, device=points.device))
    return result


class CudaGeometry:
    def __init__(self):
        self.parents = OrderedDict()
        self.placement = 'main'
        self.counters = dict(batches=0, candidate_points=0, max_candidate_points=0,
                             parent_uploads=0, seconds=0., validated_batches=0)

    @torch.no_grad()
    def materialize(self, batch):
        began = time.perf_counter()
        request = batch.pop('_vc_geometry')
        crop, requests = request['crop'], request['requests']
        shape = (crop.depth, crop.width, crop.width)
        sizes = [len(item['points']) if item is not None else 0 for item in requests]
        offsets = np.r_[0, np.cumsum(sizes)]
        n = int(offsets[-1])
        keys = torch.empty((len(requests), 32768), device='cuda', dtype=torch.int64)
        if n > keys.numel():
            raise ValueError(f'{n} candidate points exceed sparse key capacity {keys.numel()}')
        counts = torch.zeros(len(requests), device='cuda')
        ids = torch.as_tensor(np.repeat(np.arange(len(requests)), sizes), device='cuda')
        voxels = torch.empty((n, 3), device='cuda')
        if n:
            points = torch.as_tensor(np.concatenate([i['points'] for i in requests if i is not None]), device='cuda')
            frames = torch.as_tensor(np.stack([i['frame'] if i is not None else np.eye(3)
                                               for i in requests]), device='cuda')
            # Only cell quantization needs float64. Native nearest-query inputs,
            # distances, cell centers and all model tensors remain float32.
            local = torch.bmm(points[:, None], frames[ids]).squeeze(1)
            indices = torch.stack((local[:, 2]/crop.spacing+crop.behind,
                local[:, 1]/crop.spacing+(crop.width-1)/2,
                local[:, 0]/crop.spacing+(crop.width-1)/2), -1)
            near = (indices >= 0).all(-1)
            for a, size in enumerate(shape):
                near &= indices[:, a] <= size-1
            identities = [i['identity'] if i is not None else np.zeros((1, 3), np.float32) for i in requests]
            identity_offsets = np.r_[0, np.cumsum([len(p) for p in identities])]
            identity = PolylineSet(np.concatenate(identities), identity_offsets).cuda()
            local32 = local.float()
            result = nearest_outputs(local32)
            identity.nearest(local32, curve_ids=ids, vertices=True, out=result)
            bounds = torch.as_tensor(identity_offsets, device='cuda')
            keep = near & (result['indices'] > bounds[ids]) & (result['indices'] < bounds[ids+1]-1)
            voxels = indices.round().float()
            centers = torch.stack(((voxels[:, 2]-(crop.width-1)/2)*crop.spacing,
                                   (voxels[:, 1]-(crop.width-1)/2)*crop.spacing,
                                   (voxels[:, 0]-crop.behind)*crop.spacing), -1)
            positions = torch.as_tensor(np.stack([i['position'] if i is not None else np.zeros(3, np.float32)
                                                  for i in requests]), device='cuda')
            affines = torch.cat((frames.float(), positions[:, :, None]), -1)
            world_centers = torch.empty_like(centers)
            transform_points(centers, affines[ids], out=world_centers)
            queries = torch.stack((points.float()+positions[ids], world_centers), 1).reshape(-1, 3)
            parent_result = nearest_outputs(queries)
        for row, item in enumerate(requests):
            if not sizes[row]:
                continue
            key = item['key']
            if key not in self.parents:
                self.parents[key] = PolylineSet(item['parent'], [0, len(item['parent'])]).cuda()
                self.counters['parent_uploads'] += 1
                if len(self.parents) > 256:
                    self.parents.popitem(last=False)
            self.parents.move_to_end(key)
            part = slice(2*offsets[row], 2*offsets[row+1])
            self.parents[key].nearest(queries[part], vertices=True,
                                     out={k: v[part] for k, v in parent_result.items()})
        if n:
            margins = torch.as_tensor(np.array([[i['gap'], i['exclusion']] if i is not None else [0., 0.]
                                                for i in requests], np.float32), device='cuda')[ids]
            distances = parent_result['distances'].reshape(-1, 2)
            keep &= distances[:, 0]-margins[:, 0] > margins[:, 1]
            counts.scatter_reduce_(0, ids, keep.float(), reduce='amax')
            keep &= distances[:, 1]-margins[:, 0]-np.sqrt(3)*crop.spacing/2 > margins[:, 1]
            voxels = torch.where(keep[:, None], voxels, torch.nan)
        occupancy = VoxelSet(voxels, shape, batch_ids=ids, batches=len(requests))
        # Stable shape for the trainer's fullgraph compiled loss. Oversize
        # inputs raise rather than silently discarding any occupied cells.
        occupancy.sorted_keys(capacity=keys.numel(), out=keys.view(-1))
        batch['foreign'] = keys
        # These small counts feed the trainer's existing CPU metrics only.
        batch['foreign_components'] = counts.cpu()
        self.counters['candidate_points'] += len(voxels)
        self.counters['max_candidate_points'] = max(self.counters['max_candidate_points'], len(voxels))
        self.counters['batches'] += 1
        self.counters['seconds'] += time.perf_counter()-began
        if 'expected' in request:
            expected = request['expected']
            actual = torch.empty((len(requests), *shape), device='cuda', dtype=torch.uint8)
            occupancy.to_dense(out=actual)
            try:
                np.testing.assert_array_equal(actual.cpu().numpy(), expected['foreign'].numpy())
                torch.testing.assert_close(batch['foreign_components'], expected['foreign_components'])
            except AssertionError:
                path = os.environ.get('FIBER_BENCH_GEOMETRY_FAILURE')
                if path:
                    torch.save(request, path)
                raise
            self.counters['validated_batches'] += 1

    def stats(self):
        return dict(self.counters, backend='cuda', worker_planes='vc_reader CPU',
                    placement=self.placement,
                    foreign='sorted sparse keys; CUDA clearance, transforms and voxelization',
                    sampling_feedback='CPU trainer policy', resident_parent_limit=256)
