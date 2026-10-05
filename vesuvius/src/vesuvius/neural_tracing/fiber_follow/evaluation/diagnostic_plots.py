"""Contact sheets with fixed physical crop coordinates and actual activations."""
from itertools import product, combinations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps


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


def display_example(batch, output, details, layers, cfg, label, metrics):
    crop = cfg.fine
    image = array(batch['x']['fine'][0, 0])
    center = (crop.width-1)/2
    indices = [int(np.floor(center)), int(np.ceil(center))]
    # Linear interpolation at u=v=0, not a target-following curved slice.
    sections = [image[:, indices, :].mean(1), image[:, :, indices].mean(2), image[int(crop.behind)]]
    history = masked_line(array(batch['hist'][0]), array(batch['hmask'][0]) > 0)
    annotation = annotation_points(batch, cfg)
    selected = int(output['selected_refinement'][0])
    flow = 'solver_points' in output
    decoder = {key: array(values[-1 if flow else selected]) for key, values in layers['decoder'].items()}
    memory = {}
    if 'history_valid' in batch['x']:  # memory models only (the history sheet)
        valid = array(batch['x']['history_valid'][0]).astype(bool)
        tokens = array(layers['history']['tokens'])
        if 'history_slabs' in batch['x']:
            slabs = array(batch['x']['history_slabs'][0])
            # Central slab CT section and observed path heatmap; no target-based slicing.
            middle = (slabs.shape[2]-1)/2
            indices = [int(np.floor(middle)), int(np.ceil(middle))]
            ct = slabs[:, 0, indices].mean(1)
            heat = slabs[:, 1, indices].mean(1).clip(0, 1)[..., None]
            gray = ((ct+4)/8).clip(0, 1)[..., None]
            rgb = gray*(1-.7*heat)+np.array([1., .15, .15])*.7*heat
        else:
            # Decision memory has no separate memory CT; valid slots are shown blank.
            rgb = np.full((*tokens.shape, 3), .6)
        rgb[~valid] = .15
        attentions = {}
        for name, depth in layers['history_depths'].items():
            values = layers['history'].get(name+'_attention', [])
            chosen = values[-depth:] if flow and name == 'generator' else values[selected*depth:(selected+1)*depth]
            attentions[name] = np.mean([array(a) for a in chosen], axis=0) if chosen else np.zeros((cfg.n_future, len(valid)))
        convolution = np.zeros_like(tokens)
        if layers['history']['convolution'] is not None:
            convolution[valid] = array(layers['history']['convolution'])
        memory = dict(history_ct=rgb, history_conv=convolution, history_tokens=tokens, history_valid=valid,
                      attention=attentions)
    return dict(label=label, sections=sections, annotation=annotation, history=history,
        points=array(output['points'][0]), initial=array(output['initial_points'][0]),
        confidence=array(output['confidence'][0]), details=details, selected=selected,
        encoder={k: array(v) for k, v in layers['encoder'].items()}, decoder=decoder, metrics=metrics, **memory)


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


def activation_sheet(examples, step, kind):
    stages = list(examples[0][kind])
    subtitle = ('Spatial channel-contrast RMS; fixed central v section. Shared scale per stage across all examples. Not attention.'
                if kind == 'encoder' else 'Actual query x hidden-channel activations for the selected attempt. Shared signed scale per stage. Not CT pixels.')
    canvas = new_sheet(examples, len(stages), f'Step {step} | {kind} stages', subtitle)
    maxima = {stage: max(float(np.nanmax(np.abs(e[kind][stage]))) for e in examples) for stage in stages}
    for row, e in enumerate(examples):
        for col, stage in enumerate(stages):
            value = e[kind][stage]
            p = Panel(canvas, row, col, f'{e["label"]} | {stage}')
            scale = max(maxima[stage], 1e-8)
            p.image(value, lo=0 if kind == 'encoder' else -scale, hi=scale,
                    cmap='magma' if kind == 'encoder' else 'RdBu_r')
            p.axes('u feature index' if kind == 'encoder' else 'hidden channel',
                   'forward feature index' if kind == 'encoder' else f'forecast query; attempt {e["selected"]}')
    return canvas


def mosaic(slots):
    # origin is lower: slots 0-3 are on the bottom row, 4-7 on the top row.
    return np.concatenate([np.concatenate(list(slots[start:start+4]), axis=1) for start in (0, 4)], axis=0)


def history_sheet(examples, cfg, step):
    canvas = new_sheet(examples, 5, f'Step {step} | history CT -> convolution -> tokens -> attention',
        'Slots: top 4-7, bottom 0-3. Padding gray/zero. Attention is head/layer mean for selected attempt; not causal attribution.')
    limits = {key: max(float(np.nanmax(e[key])) for e in examples) for key in ('history_conv', 'history_tokens')}
    for row, e in enumerate(examples):
        for col, key in enumerate(('history_ct', 'history_conv', 'history_tokens')):
            p = Panel(canvas, row, col, e['label']+' | '+('CT + path', 'conv RMS', 'token contrast')[col])
            p.image(mosaic(e[key]), hi=1 if col == 0 else max(limits[key], 1e-8), equal=True)
            p.draw.text((p.rect[0], p.rect[3]+4), 'valid slots: '+','.join(str(i) for i, valid in enumerate(e['history_valid']) if valid), font=SMALL, fill='#444444')
        for col, head in enumerate(('generator', 'scorer'), 3):
            p = Panel(canvas, row, col, e['label']+' | '+head+' attention', ((0, 7), (0, cfg.n_future-1)))
            p.image(e['attention'][head], hi=1, cmap='viridis')
            p.axes('history slot', 'forecast query')
    return canvas


def plot_sheets(examples, cfg, folder, step):
    if not examples:
        raise ValueError('Diagnostic microbatch is empty')
    sheets = dict(predictions=predictions_sheet(examples, cfg, step),
        crop_orientation=orientation_sheet(examples, cfg, step),
        encoder=activation_sheet(examples, step, 'encoder'), decoder=activation_sheet(examples, step, 'decoder'),
        **({'history': history_sheet(examples, cfg, step)} if 'history_valid' in examples[0] else {}))
    # Independent PNG compression/writes run together; keep compression cheap.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(image.save, folder/(name+'.png'), compress_level=1) for name, image in sheets.items()]
        for future in futures:
            future.result()
    for image in sheets.values():
        image.close()
