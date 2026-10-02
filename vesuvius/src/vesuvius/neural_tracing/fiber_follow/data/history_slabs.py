"""Live causal observations. No annotation, cached feature, or writer inputs."""
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import (
    CropSpec, arclength, interp_at, frame_from_heading, normalize, crop_local_grid,
    render_history,
)

from vesuvius.neural_tracing.fiber_follow.shared.reference import observed_path
from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_frame, reframe_item, FRAME_POLICY

SLOTS = 8
SLAB = CropSpec(depth=8, width=65, behind=4, spacing=.5)
SAMPLING_REVISION = 'live_observed_slabs_ct_transverse_v3'
PATH_SAMPLES = 3


def slab_path_samples(item, slab):
    """Sample the actual committed polyline near this observation, never GT."""
    path = observed_path(item)
    arc = arclength(path)
    at = arc[-1]-slab['age']+np.array([-1., 0., 1.])
    points = interp_at(path, arc, np.clip(at, 0., arc[-1]))
    before = interp_at(path, arc, np.clip(at-.25, 0., arc[-1]))
    after = interp_at(path, arc, np.clip(at+.25, 0., arc[-1]))
    tangent = (after-before) @ slab['frame']
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = tangent/np.maximum(norm, 1e-8)
    points = (points-slab['pos']) @ slab['frame']
    lo = np.array([-16., -16., -2.])
    hi = np.array([16., 16., 1.5])
    valid = (at >= -1e-8) & (at <= arc[-1]+1e-8) & (norm[:, 0] > 1e-8)
    valid &= ((points >= lo-1e-6) & (points <= hi+1e-6)).all(-1)
    return points, tangent, valid




def selected_arcs(length):
    """Seed, up to four older samples, then available 96/64/32 anchors."""
    older = max(0., length-96.)
    count = min(4, max(0, int(np.floor(older/32.))-1))
    values = [0.]
    if count:
        values.extend(older*np.arange(1, count+1)/(count+1))
    for age in (96., 64., 32.):
        at = length-age
        if at-values[-1] >= 32.-1e-8:
            values.append(at)
    return np.asarray(values)


def fitted_heading(path, arc, at, seed_heading):
    """Four points, two voxels apart, regressed against traversal arclength."""
    if arc[-1] >= 6.:
        start = np.clip(at-3., 0., arc[-1]-6.)
        points = interp_at(path, arc, start+np.arange(4)*2.)
        heading = (points*np.array([-3., -1., 1., 3.])[:, None]).sum(0)
        if np.isfinite(heading).all() and np.linalg.norm(heading) > 1e-8:
            return normalize(heading)
    return normalize(np.asarray(seed_heading, dtype=np.float64))


def slab_layout(item):
    path = observed_path(item)
    arc = arclength(path)
    heading = item['seed_tangent'] if item.get('seed_valid', False) else np.asarray(item['frame'])[:, 2]
    if not np.isfinite(heading).all() or np.linalg.norm(heading) < 1e-8:
        raise ValueError('A finite nonzero seed heading is required')
    slabs = []
    for slot, at in enumerate(selected_arcs(arc[-1])):
        pos = interp_at(path, arc, np.array([at]))[0]
        frame = frame_from_heading(fitted_heading(path, arc, at, heading))
        lo, hi = max(0., at-8.), min(arc[-1], at+8.)
        # Include exact boundaries and every committed vertex between them.
        local_arc = np.unique(np.r_[lo, arc[(arc > lo) & (arc < hi)], at, hi])
        local = (interp_at(path, arc, local_arc)-pos) @ frame
        # Renderer joins origin to its first valid point. A masked leading
        # point breaks that artificial chord; real segments remain connected.
        local = np.concatenate((np.zeros((1, 3)), local))
        slabs.append(dict(pos=pos, frame=frame, hist_local=local,
                          hmask=np.r_[0., np.ones(len(local)-1)],
                          age=arc[-1]-at, seed=slot == 0))
    return slabs


def slabs_allowed(item, band):
    from vesuvius.neural_tracing.fiber_follow.data.data import training_state_allowed
    return all(training_state_allowed(slab, SLAB, band) for slab in slab_layout(item))


def load_slabs(items, vol, cfg, pool=None):
    from vesuvius.neural_tracing.fiber_follow.data.observations import visible_points
    from vesuvius.neural_tracing.fiber_follow.data.crop_sampling import scalar_crops, empty_image_batch
    started = time.perf_counter()
    def orient(item):
        # Frames chain within an item (sign continuity), never across items.
        layout = slab_layout(item)
        previous = None
        for slab in layout:
            diagnostics = {}
            fallback = item['frame'] if item.get('frame_policy') == FRAME_POLICY else None
            frame = ct_frame(vol, slab['pos'], slab['frame'][:, 2], previous,
                             fallback=fallback, diagnostics=diagnostics)
            reframe_item(slab, frame)
            slab['frame_policy'] = FRAME_POLICY
            slab['ct_frame_diagnostics'] = diagnostics
            previous = frame
        item['_sampled_slabs'] = layout
        return layout
    layouts = list(map(orient, items) if pool is None else pool.map(orient, items))
    flat = [slab for layout in layouts for slab in layout]
    images = scalar_crops(flat, vol, SLAB, pool, presence=False)
    grid = torch.from_numpy(crop_local_grid(SLAB)).float()
    output = empty_image_batch((len(items), SLOTS, 2, 8, 65, 65)).zero_()
    valid = torch.zeros(len(items), SLOTS, dtype=torch.bool)
    # Translation (3), relative rotation (9), log age (1), seed role (1).
    pose = torch.zeros(len(items), SLOTS, 14)
    ages, overlap = torch.zeros(len(items), SLOTS), torch.zeros(len(items), SLOTS)
    path_inputs = {}
    path_inputs = dict(history_path_points=torch.zeros(len(items), SLOTS, PATH_SAMPLES, 3),
                       history_path_tangents=torch.zeros(len(items), SLOTS, PATH_SAMPLES, 3),
                       history_path_valid=torch.zeros(len(items), SLOTS, PATH_SAMPLES, dtype=torch.bool))
    frame_source = torch.full((len(items), SLOTS), -1, dtype=torch.int64)
    frame_energy, frame_gap = torch.zeros(len(items), SLOTS), torch.zeros(len(items), SLOTS)
    offsets = np.cumsum([0]+[len(layout) for layout in layouts])

    inference = torch.is_inference_mode_enabled()

    def fill(row):
        # Writes only this row of each preallocated tensor. Pool threads do not inherit
        # the caller's inference mode, which in-place writes to its tensors require.
        with torch.inference_mode(inference):
            item, layout = items[row], layouts[row]
            for slot, slab in enumerate(layout):
                cursor = int(offsets[row])+slot
                for key, value in zip(path_inputs, slab_path_samples(item, slab)):
                    path_inputs[key][row, slot] = torch.as_tensor(value)
                heat = render_history(torch.as_tensor(slab['hist_local'])[None].float(),
                                      torch.as_tensor(slab['hmask'])[None], grid, 1., 'segments')[0]
                output[row, slot] = torch.cat((images[cursor], heat), 0)
                valid[row, slot] = True
                quality = slab['ct_frame_diagnostics']
                frame_source[row, slot] = quality['source']
                frame_energy[row, slot], frame_gap[row, slot] = quality['energy'], quality['gap']
                relative = (slab['pos']-item['pos']) @ item['frame']
                rotation = np.asarray(item['frame']).T @ slab['frame']
                pose[row, slot] = torch.tensor(np.r_[relative/128., rotation.ravel(),
                                                       np.log1p(slab['age'])/8., float(slab['seed'])])
                ages[row, slot] = slab['age']
                world = grid.numpy().reshape(-1, 3) @ slab['frame'].T+slab['pos']
                overlap[row, slot] = float(visible_points((world-item['pos']) @ item['frame'], cfg.fine).mean())
    if pool is None:
        for row in range(len(items)):
            fill(row)
    else:
        list(pool.map(fill, range(len(items))))
    elapsed = torch.full((len(items),), (time.perf_counter()-started)/max(1, len(items)))
    return dict(history_slabs=output, history_valid=valid, history_pose=pose,
                history_ages=ages, history_overlap=overlap, history_load_seconds=elapsed,
                history_frame_source=frame_source, history_frame_energy=frame_energy, history_frame_gap=frame_gap,
                **path_inputs)


