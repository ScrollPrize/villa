"""Current-microbatch contact sheets (fixed physical crop coordinates, actual activations) and cheap measurements;
PNG/JSON only."""
from itertools import product, combinations
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.tracers.fibers.data.labels import prefix_labels
from vesuvius.neural_tracing.tracers.shared.policy import commit_prefix
from vesuvius.neural_tracing.tracers.models.targets import geometry_mask, foreign_failures


def slice_batch(batch, selection):
    return {k: slice_batch(v, selection) if isinstance(v, dict) else v[selection]
            for k, v in batch.items()}


def cpu_values(value):
    if isinstance(value, dict):
        return {k: cpu_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_values(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu()
    return value


def tensor_stats(value):
    value = value.detach().float()
    keys = ('finite_fraction', 'rms', 'mean', 'std', 'abs_max', 'zero_fraction')
    if not value.numel():
        return dict(count=0, **dict.fromkeys(keys))
    finite = torch.isfinite(value)
    clean = torch.where(finite, value, 0.)
    count = finite.sum().clamp_min(1)
    mean = clean.sum()/count
    variance = torch.where(finite, (clean-mean).square(), 0.).sum()/count
    # One small host transfer, avoiding per-statistic CUDA synchronization and
    # dynamic boolean indexing of full spatial activation banks.
    values = torch.stack((finite.float().mean(), (clean.square().sum()/count).sqrt(),
                          mean, variance.sqrt(), clean.abs().max(),
                          ((clean == 0) & finite).sum()/count)).cpu().tolist()
    if values[0] == 0:
        values[1:] = [None]*5
    return dict(count=value.numel(), **dict(zip(keys, values)))


def decision_details(output, batch, cfg, n_commit, tolerance):
    """Use training's identity-aware labels and the actual deployed commit policy."""
    curve = output['points']
    mask = geometry_mask(batch, cfg)
    def errors(points):
        dense = F.interpolate(points[..., :2].transpose(1, 2), size=mask.shape[1],
                              mode='linear', align_corners=True).transpose(1, 2)
        return (dense-batch['dense_ab']).norm(dim=-1).masked_fill(~mask, float('nan'))
    foreign = foreign_failures(curve, batch, cfg, mask.shape[1]) if 'foreign' in batch else None
    labels, known, _ = prefix_labels(curve, batch, tolerance, cfg.max_recovery_distance, extra_failure=foreign)
    if 'identity_observable' in batch:
        known *= batch['identity_observable'][:, None]
    counts, allowed = commit_prefix(curve, output['confidence'], .5, n_commit, cfg.max_recovery_distance)
    attempts = []
    for i, proposal in enumerate(output['refinement_points'].unbind(1)):
        count, _ = commit_prefix(proposal, output['refinement_confidence'][:, i], .5, n_commit, cfg.max_recovery_distance)
        error = errors(proposal)[0]
        valid = torch.isfinite(error)
        attempts.append(dict(attempt=i, attempted=bool(output['refinement_mask'][0, i]),
                             selected=int(output['selected_refinement'][0]) == i,
                             error=float(error[valid].mean()) if valid.any() else None,
                             commit=int(count[0])))
    frame_quality = {key: value[0] for key, value in batch['x'].items()
                     if key.startswith('ct_frame_')}
    return cpu_values(dict(labels=labels[0], known=known[0], error=errors(curve)[0],
        initial_error=errors(output['initial_points'])[0], commit=int(counts[0]),
        connection_allowed=bool(allowed[0]), attempts=attempts, frame_quality=frame_quality))


def json_values(value):
    """Unknown or nonfinite measurements stay null in strict JSON."""
    import math
    if torch.is_tensor(value):
        return json_values(value.tolist())
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value



def measured_geometry(batch):
    """Geometry is in the same local frame as the sampled CT, without warping."""
    annotation = batch.get('diagnostic_annotation')
    valid = batch.get('diagnostic_annotation_mask')
    angle = None
    if annotation is not None:
        points = annotation[0][valid[0]].float()
        if len(points) >= 2:
            near = int(points.square().sum(-1).argmin())
            tangent = points[min(len(points)-1, near+2)]-points[max(0, near-2)]
            if tangent.norm() > 1e-8:
                angle = float(torch.rad2deg(torch.atan2(tangent[:2].norm(), tangent[2])))
    return dict(heading_annotation_angle_degrees=angle,
                position_xyz=cpu_values(batch['crop_pos'][0]) if 'crop_pos' in batch else None,
                frame_columns_uv_heading_xyz=cpu_values(batch['crop_frame'][0]) if 'crop_frame' in batch else None)


SEQUENCE_EPISODES, SEQUENCE_DECISIONS = 4, 4  # sequence sheets: the last decisions of the first episodes


@torch.no_grad()
def sequence_rows(model, cpu_batch, device, n_commit):
    """A sequence model's episode batch reduced to the rows the sheets show: the last ``SEQUENCE_DECISIONS``
    supervised decisions of the first ``SEQUENCE_EPISODES`` episodes, each predicted exactly as in training (its own
    crop with its episode's earlier steps as history, train/sequence.py). Returns those rows and their outputs."""
    from vesuvius.neural_tracing.tracers.fibers.train.sequence import episode_forward, select_rows
    from vesuvius.neural_tracing.tracers.shared.tensors import move_batch
    from vesuvius.neural_tracing.tracers.shared.policy import DEFAULT_CONFIDENCE
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
        output, _ = episode_forward(model, move_batch(cpu_batch, device), DEFAULT_CONFIDENCE, n_commit)
    supervised = cpu_batch['episode_supervised'].bool().nonzero().flatten()
    episodes = cpu_batch['episode_index'][supervised]
    picks = [k for episode in episodes.unique(sorted=True)[:SEQUENCE_EPISODES].tolist()
             for k in (episodes == episode).nonzero().flatten()[-SEQUENCE_DECISIONS:].tolist()]
    picks = torch.tensor(picks, dtype=torch.long)
    count = len(supervised)
    outputs = {key: value[picks.to(value.device)] for key, value in output.items()
               if torch.is_tensor(value) and value.ndim and len(value) == count}
    return select_rows(cpu_batch, supervised[picks], len(cpu_batch['hist'])), outputs


@torch.no_grad()
def render_microbatch(model, cpu_batch, out, step, *, device, n_commit, tolerance,
                      dataset_names=(), training_metrics=None):
    """One sheet per image type; every current-microbatch row, in loader order."""
    from vesuvius.neural_tracing.tracers.shared.tensors import move_batch
    folder = Path(out)/'diagnostic_images'/str(step)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    examples, rows = [], []
    inference_seconds = 0.
    try:
        episode_outputs = None
        if model.model_type == 'sequence':
            cpu_batch, episode_outputs = sequence_rows(model, cpu_batch, device, n_commit)
        for i in range(len(cpu_batch['hist'])):
            batch = move_batch(slice_batch(cpu_batch, slice(i, i+1)), device)
            # Singleton inference keeps each adaptive refinement's row identity
            # unambiguous, and bounds peak diagnostic GPU memory.
            begin = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
                if episode_outputs is None:
                    output = model(batch['x'], batch['hist'], batch['hmask'], n_commit=n_commit)
                else:  # a sequence decision, predicted with its episode's history (sequence_rows)
                    output = {key: value[i:i+1] for key, value in episode_outputs.items()}
            details = decision_details(output, batch, model.cfg, n_commit, tolerance)
            output, batch = cpu_values(output), cpu_values(batch)
            inference_seconds += time.perf_counter()-begin  # CPU transfer above synchronizes CUDA.
            dataset_id = int(batch['dataset_id'][0]) if 'dataset_id' in batch else 0
            name = dataset_names[dataset_id] if 0 <= dataset_id < len(dataset_names) else str(dataset_id)
            label = f'{i:02d} | {name}'
            selected = int(output['selected_refinement'][0])
            known = details['known'].bool()
            errors = details['error'][torch.isfinite(details['error'])]
            confidence = output['confidence'][0]
            row = dict(example=i, dataset=name, dataset_id=dataset_id,
                source=int(batch['source'][0]) if 'source' in batch else None,
                terminal=bool(batch['terminal'][0]), supervision=int(batch['supervision'][0]),
                identity_observable=bool(batch['identity_observable'][0]) if 'identity_observable' in batch else None,
                selected_attempt=selected, **details, **measured_geometry(batch),
                confidence=confidence, hazard_probability=output['hazard_logits'][0].sigmoid(),
                mean_error=float(errors.mean()) if len(errors) else None,
                p95_error=float(errors.quantile(.95)) if len(errors) else None,
                max_error=float(errors.max()) if len(errors) else None,
                known_prefixes=int(known.sum()),
                prefix_brier=float((confidence[known]-details['labels'][known]).square().mean()) if known.any() else None,
                ct_statistics=tensor_stats(batch['x']['fine'][:, 0]))
            rows.append(row)
            # Retain only display-sized CT sections and reduced features on CPU.
            examples.append(display_example(batch, output, details, model.cfg, label, row))
        plot_sheets(examples, model.cfg, folder, step)
        errors = [r['mean_error'] for r in rows if r['mean_error'] is not None]
        report = dict(step=step, split='current_training_microbatch', model='EMA', model_type=model.model_type, examples=len(rows),
            n_commit=n_commit, tolerance=tolerance,
            summary=dict(committed_points=sum(r['commit'] for r in rows),
                stopped=sum(r['commit'] == 0 for r in rows),
                mean_example_error=sum(errors)/len(errors) if errors else None,
                geometry_examples=len(errors)),
            timing=dict(inference_and_measurement_seconds=inference_seconds,
                        total_diagnostic_seconds=time.perf_counter()-started),
            training_update=training_metrics or {}, rows=rows,
            interpretation=dict(orientation='Fixed CT sections and annotation in the actual crop frame; no GT-following reslicing.',
                confidence='Prefix survival; unknown labels excluded from error and calibration metrics.',
                sampling='Current augmented training microbatch; not held-out validation.'))
        (folder/'metrics.json').write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+'\n')
        report['timing']['total_diagnostic_seconds'] = time.perf_counter()-started
        return report['summary'] | dict(examples=len(rows), **report['timing'])
    finally:
        model.train(was_training)


def array(value):
    return value.detach().float().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def masked_line(points, mask):
    result = np.array(points, dtype=float, copy=True)
    result[~np.asarray(mask, dtype=bool)] = np.nan
    return result


def annotation_points(batch, cfg):
    if 'diagnostic_annotation' in batch:
        return masked_line(array(batch['diagnostic_annotation'][0]), array(batch['diagnostic_annotation_mask'][0]) > 0)
    # Tests and externally constructed observations may supply only plane labels.
    if 'plane_ab' in batch:
        points = np.c_[array(batch['plane_ab'][0]), np.arange(1, cfg.n_future+1)*cfg.future_step]
        return masked_line(points, array(batch['plane_mask'][0]) > 0)
    if 'dense_ab' in batch:
        points = np.c_[array(batch['dense_ab'][0]), np.linspace(cfg.future_step, cfg.n_future*cfg.future_step, batch['dense_ab'].shape[1])]
        return masked_line(points, array(batch['dense_mask'][0]) > 0)
    return np.empty((0, 3))


def display_example(batch, output, details, cfg, label, metrics):
    crop = cfg.fine
    image = array(batch['x']['fine'][0, 0])
    center = (crop.width-1)/2
    indices = [int(np.floor(center)), int(np.ceil(center))]
    # Linear interpolation at u=v=0, not a target-following curved slice.
    sections = [image[:, indices, :].mean(1), image[:, :, indices].mean(2), image[int(crop.behind)]]
    history = masked_line(array(batch['hist'][0]), array(batch['hmask'][0]) > 0)
    annotation = annotation_points(batch, cfg)
    selected = int(output['selected_refinement'][0])
    return dict(label=label, sections=sections, annotation=annotation, history=history,
        points=array(output['points'][0]), initial=array(output['initial_points'][0]),
        confidence=array(output['confidence'][0]), details=details, selected=selected, metrics=metrics)


def bounds(crop):
    half = crop.width*crop.spacing/2
    return (-half, half), ((-.5-crop.behind)*crop.spacing, (crop.depth-.5-crop.behind)*crop.spacing)


def clipped_polyline(points, lower, upper):
    """Clip 3D segments to a box, retaining crossings and breaking hidden gaps."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.empty((0, 3))
    start, delta = points[:-1], np.diff(points, axis=0)
    parallel = delta == 0
    denominator = np.where(parallel, 1., delta)
    t0, t1 = (np.asarray(lower)-start)/denominator, (np.asarray(upper)-start)/denominator
    enter = np.maximum(0., np.where(parallel, -np.inf, np.minimum(t0, t1)).max(axis=1))
    leave = np.minimum(1., np.where(parallel, np.inf, np.maximum(t0, t1)).min(axis=1))
    visible = ((enter <= leave) & np.isfinite(start).all(axis=1) & np.isfinite(delta).all(axis=1)
               & ~((parallel & ((start < lower) | (start > upper))).any(axis=1)))
    result = np.full((int(visible.sum()), 3, 3), np.nan)
    result[:, 0] = start[visible]+enter[visible, None]*delta[visible]
    result[:, 1] = start[visible]+leave[visible, None]*delta[visible]
    return result.reshape(-1, 3)


# Direct raster drawing avoids constructing hundreds of Matplotlib axes at each
# event. Every panel is a scientific array/curve view with explicit coordinates.
FONT = ImageFont.load_default(size=13)


SMALL = ImageFont.load_default(size=11)


CELL = (340, 250)


def new_sheet(examples, columns, title, subtitle):
    canvas = Image.new('RGB', (columns*CELL[0], 62+len(examples)*CELL[1]), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=FONT, fill='black')
    draw.text((12, 30), subtitle, font=SMALL, fill='#444444')
    return canvas


class Panel:
    def __init__(self, canvas, row, col, title, limits=None):
        self.canvas = canvas
        x, y = col*CELL[0], 62+row*CELL[1]
        self.rect = (x+40, y+37, x+CELL[0]-18, y+CELL[1]-36)
        self.draw = ImageDraw.Draw(canvas)
        self.draw.text((x+8, y+7), title, fill='black', font=SMALL)
        self.limits = limits

    def image(self, values, *, lo=0., hi=1., cmap='magma', equal=False):
        values = np.asarray(values)
        if values.ndim == 2:
            norm = np.nan_to_num((values-lo)/max(hi-lo, 1e-12)).clip(0, 1)
            pixels = colormaps[cmap](norm, bytes=True)[..., :3]
        else:
            pixels = (np.nan_to_num(values).clip(0, 1)*255).astype(np.uint8)
        x0, y0, x1, y1 = self.rect
        width, height = x1-x0, y1-y0
        if equal:
            ratio = min(width/pixels.shape[1], height/pixels.shape[0])
            w, h = round(pixels.shape[1]*ratio), round(pixels.shape[0]*ratio)
            x0 += (width-w)//2; x1 = x0+w
            y0 += (height-h)//2; y1 = y0+h
            self.rect = (x0, y0, x1, y1)
        image = Image.fromarray(pixels[::-1]).resize((x1-x0, y1-y0), Image.Resampling.NEAREST)
        self.canvas.paste(image, (x0, y0))
        if values.ndim == 2:
            self.draw.text((x0, y1+17), f'scale {lo:.3g} .. {hi:.3g}', fill='#555555', font=SMALL)

    def coords(self, points):
        x0, y0, x1, y1 = self.rect
        (lo, hi), (bottom, top) = self.limits
        p = np.asarray(points)
        return np.c_[x0+(p[:, 0]-lo)/(hi-lo)*(x1-x0-1), y1-1-(p[:, 1]-bottom)/(top-bottom)*(y1-y0-1)]

    def line(self, points, color, width=1, dots=False):
        if self.limits is None or len(points) == 0:
            return
        # Render to a clipped overlay: out-of-crop paths cannot paint other rows.
        x0, y0, x1, y1 = self.rect
        overlay = Image.new('RGBA', (x1-x0, y1-y0))
        draw = ImageDraw.Draw(overlay)
        xy = self.coords(points)-[x0, y0]
        last = None
        for i, point in enumerate(xy):
            if not np.isfinite(point).all():
                last = None
                continue
            point = tuple(point)
            if last is not None and (not dots or i % 2):
                draw.line([last, point], fill=color, width=width)
            last = point
        self.canvas.paste(overlay, (x0, y0), overlay)

    def axes(self, xlabel, ylabel):
        x0, y0, x1, y1 = self.rect
        self.draw.rectangle(self.rect, outline='#777777')
        self.draw.text((x0, y0-13), ylabel, fill='#444444', font=SMALL)
        if self.limits:
            (lo, hi), (bottom, top) = self.limits
            for fraction in (0, .5, 1):
                x = x0+fraction*(x1-x0)
                self.draw.text((x-8, y1+2), f'{lo+fraction*(hi-lo):g}', fill='#444444', font=SMALL)
            self.draw.text((x0-34, y0), f'{top:g}', fill='#444444', font=SMALL)
            self.draw.text((x0-34, y1-12), f'{bottom:g}', fill='#444444', font=SMALL)
        self.draw.text((x0+90, y1+17), xlabel, fill='#444444', font=SMALL)


def ct_panel(canvas, row, col, example, cfg, component, title):
    lateral, forward = bounds(cfg.fine)
    panel = Panel(canvas, row, col, example['label']+' | '+title,
                  (lateral, forward if component < 2 else lateral))
    panel.image(example['sections'][component], lo=-4, hi=4, cmap='gray', equal=True)
    return panel


def predictions_sheet(examples, cfg, step):
    canvas = new_sheet(examples, 3, f'Step {step} | current training microbatch | EMA predictions',
        'Lime: annotation. Red: observed history. Cyan: initial. Orange: selected, thick = committed. Fixed central CT sections [-4,4].')
    for row, e in enumerate(examples):
        for c in range(2):
            p = ct_panel(canvas, row, c, e, cfg, c, ('u/f', 'v/f')[c]+' projection')
            for data, color, width in ((e['annotation'], '#80ff00', 2), (e['history'], '#ff4444', 2),
                                        (e['initial'], '#00ccff', 1), (e['points'], '#ffb040', 1)):
                p.line(data[:, [c, 2]], color, width)
            committed = np.r_[np.zeros((1, 3)), e['points'][:e['details']['commit']]]
            p.line(committed[:, [c, 2]], '#ffb040', 3)
            p.axes(('u', 'v')[c]+' (vox)', 'forward (vox)')
        p = Panel(canvas, row, 2, f'{e["label"]} | commit {e["details"]["commit"]}/{cfg.n_future}, attempt {e["selected"]}',
                  ((0, cfg.n_future*cfg.future_step), (-.15, 1.)))
        p.line([[0, .5], [cfg.n_future*cfg.future_step, .5]], '#777777', dots=True)
        p.line(np.c_[e['points'][:, 2], e['confidence']], '#3366bb', 2)
        known, labels = array(e['details']['known']), array(e['details']['labels'])
        for x, k, label in zip(e['points'][:, 2], known, labels):
            p.line([[x, -.11], [x, -.04]], '#338833' if k and label else '#dd4444' if k else '#999999', 4)
        p.axes('forward (vox)', 'prefix survival; GT strip: green/red/gray')
    return canvas


def orientation_sheet(examples, cfg, step):
    canvas = new_sheet(examples, 4, f'Step {step} | actual crop orientation versus annotation',
        'Green: annotation within half a crop voxel of each fixed section. Blue dot: crop origin. Full in-crop path at right; h = heading.')
    lateral, forward = bounds(cfg.fine)
    lower = np.array([lateral[0], lateral[0], forward[0]])
    upper = np.array([lateral[1], lateral[1], forward[1]])
    corners = np.array(list(product(lateral, lateral, forward)))
    # Fixed orthographic camera: all examples share the same crop-space view.
    camera = np.array([[.8, -.6, 0.], [.25, .33, .91]])
    box = corners @ camera.T
    limits = tuple((float(box[:, i].min())-2, float(box[:, i].max())+2) for i in range(2))
    for row, e in enumerate(examples):
        a = e['annotation']
        for c, (x, y, hidden) in enumerate(((0, 2, 1), (1, 2, 0), (0, 1, 2))):
            p = ct_panel(canvas, row, c, e, cfg, c, ('v=0', 'u=0', 'f=0')[c])
            lo, hi = lower.copy(), upper.copy()
            lo[hidden], hi[hidden] = -cfg.fine.spacing/2, cfg.fine.spacing/2
            near = clipped_polyline(a, lo, hi)
            p.line(near[:, [x, y]], '#80ff00', 2)
            # Heading points out of the f=0 plane; a vertical line there
            # would incorrectly identify the v axis as the heading.
            ox, oy = p.coords([[0, 0]])[0]
            p.draw.ellipse((ox-2, oy-2, ox+2, oy+2), fill='#3377aa')
            p.axes(('u', 'v', 'u')[c]+' (vox)', ('forward', 'forward', 'v')[c]+' (vox)')
        angle = e['metrics']['heading_annotation_angle_degrees']
        title = f'{e["label"]} | heading/GT '+(f'{angle:.1f} deg' if angle is not None else 'unknown')
        p = Panel(canvas, row, 3, title, limits)
        # Equal physical scale in the orthographic camera plane.
        (lo, hi), (bottom, top) = p.limits
        aspect = (p.rect[2]-p.rect[0])/(p.rect[3]-p.rect[1])
        if (hi-lo)/(top-bottom) < aspect:
            center, half = (lo+hi)/2, (top-bottom)*aspect/2
            p.limits = ((center-half, center+half), (bottom, top))
        else:
            center, half = (bottom+top)/2, (hi-lo)/aspect/2
            p.limits = ((lo, hi), (center-half, center+half))
        for i, j in combinations(range(8), 2):
            if np.count_nonzero(corners[i] != corners[j]) == 1:
                p.line(box[[i, j]], '#aaaaaa')
        for points, color in ((a, '#338833'), (e['history'], '#dd4444')):
            p.line(clipped_polyline(points, lower, upper) @ camera.T, color, 2)
        for axis, color, name in ((0, '#aa55aa', 'u'), (1, '#bb7700', 'v'), (2, '#2266dd', 'h')):
            tip = np.eye(3)[axis]*min(10, forward[1])
            line = np.array([np.zeros(3), tip]) @ camera.T
            p.line(line, color, 2)
            xy = p.coords(line)[-1]
            p.draw.text(tuple(xy), name, fill=color, font=SMALL)
        p.draw.text((p.rect[0], p.rect[3]+5), 'True crop box + annotation (orthographic)', fill='#444444', font=SMALL)
    return canvas


def plot_sheets(examples, cfg, folder, step):
    if not examples:
        raise ValueError('Diagnostic microbatch is empty')
    sheets = dict(predictions=predictions_sheet(examples, cfg, step),
        crop_orientation=orientation_sheet(examples, cfg, step))
    # Independent PNG compression/writes run together; keep compression cheap.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(image.save, folder/(name+'.png'), compress_level=1) for name, image in sheets.items()]
        for future in futures:
            future.result()
    for image in sheets.values():
        image.close()
