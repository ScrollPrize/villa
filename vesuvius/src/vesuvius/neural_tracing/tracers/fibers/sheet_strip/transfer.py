"""Start a single-lattice sheet-strip model from a two-level (multiscale) one: the fine CNN runs over the whole crop
(``multiscale_fine_width`` None), there is no coarse CNN, and the fine CNN may be wider (``cnn_channels``) and end in
a context block (``cnn_context_channels``).

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.transfer --config RUN.json --out START.pt \\
        [--set cnn_channels=[12,24,64] cnn_context_channels=64] [--batches 8]

``--config`` is the source run's configuration (its data and its last.pt, or ``--checkpoint``). What carries over:

- the fine CNN as it is; added channels start from fresh filters whose weights into the existing channels, and the
  weights of every layer reading the CNN features from them, are zero, so the existing features are unchanged;
- the inner polar tokens' projection, which the outer tokens (formerly on the coarse lattice) now use too;
- every layer reading CNN features at points (strip query and refinement, references, the path head): the fine
  feature and support columns; the coarse feature columns are dropped and their mean contribution over ``--batches``
  training batches is added to the layer's bias, so the layer starts close to the source's output;
- everything else unchanged.

The context block starts without influence. START.pt is a sheet-strip checkpoint of the new configuration (model and
EMA alike); its ``new_params`` (the context block) train at the run's ``new_lr_scale``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import crop_orientation, load_umbilicus
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import SheetStripConfig, SheetStripFollower
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.train import (
    REFINE_ALL, RunConfig, build_data, load_strip_model, with_tube)
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
from vesuvius.neural_tracing.tracers.shared.tensors import move_batch

# Modules whose first Linear reads CNN evidence at points: per point the fine features, the coarse ones and, unless
# noted, the crop support, then other inputs.
POINT_READERS = dict(strip_query=True, strip_refinement=True, reference_token=False, score=True, query=True,
                     refinement_fusion=True)
TOKEN_PROJECTIONS = ('cell_token.sector_projection', 'cell_token.tube_projection')  # per point: fine, support


def point_columns(points, fine, coarse, support, total):
    """Column indices of a point reader's ``total`` inputs (per point ``fine`` features, ``coarse`` ones and, with
    ``support``, the crop support; then the rest): per point the fine and coarse columns, the support columns and the
    rest."""
    width = fine+coarse+int(support)
    rest = total-points*width
    if rest < 0:
        raise ValueError(f'{total} inputs do not hold {points} points of {width}')
    blocks = [range(p*width, (p+1)*width) for p in range(points)]
    return ([list(b[:fine]) for b in blocks], [list(b[fine:fine+coarse]) for b in blocks],
            [b[-1] for b in blocks] if support else [], list(range(points*width, total)))


def map_columns(old, new, points, fine, coarse, support, mean=None):
    """The new first-layer (weight, bias) of a point reader from the old ``(weight, bias)``: fine columns kept (added
    features zero), coarse columns dropped (``mean``: the old layer's mean input, folded into the bias), support and
    the remaining columns kept."""
    w_old, b_old = old
    w_new, b_new = new
    f_cols, c_cols, s_cols, rest = point_columns(points, fine, coarse, support, w_old.shape[1])
    new_fine = (w_new.shape[1]-len(rest))//points-int(support)
    if points*(new_fine+int(support))+len(rest) != w_new.shape[1] or new_fine < fine:
        raise ValueError(f'new layer of {w_new.shape[1]} inputs does not fit {points} points of >= {fine} features')
    w = torch.zeros_like(w_new)
    width = new_fine+int(support)
    for p in range(points):
        w[:, p*width:p*width+fine] = w_old[:, f_cols[p]]
        if support:
            w[:, p*width+new_fine] = w_old[:, s_cols[p]]
    w[:, points*width:] = w_old[:, rest]
    b = b_old.clone()
    dropped = [c for cols in c_cols for c in cols]
    if dropped and mean is not None:
        b = b+w_old[:, dropped].float()@mean[dropped].float().to(w_old.device)
    return w, b


def widen(old, new):
    """``new`` (a freshly initialized tensor) holding ``old`` in its leading block: a widened convolution's added
    output channels keep their fresh filters, and its weights from added input channels into existing outputs are zero
    (vectors: the leading entries)."""
    out = new.clone()
    if old.ndim == 1:
        out[:len(old)] = old
        return out
    o, i = old.shape[:2]
    out[:o, :i] = old
    out[:o, i:] = 0
    return out


@torch.no_grad()
def mean_inputs(model, batches, umbilicus, spec, device):
    """Mean input of each point reader's first Linear over ``batches`` (the source model's training forward)."""
    sums, counts, hooks = {}, {}, []

    def hook(name):
        def record(module, inputs, output):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            sums[name] = sums.get(name, 0)+x.sum(0)
            counts[name] = counts.get(name, 0)+len(x)
        return record

    for name in POINT_READERS:
        hooks.append(getattr(model, name)[0].register_forward_hook(hook(name)))
    for batch in batches:
        batch = move_batch(batch, device)
        if umbilicus is not None and 'crop_orientation' not in batch:
            batch['crop_orientation'] = torch.from_numpy(crop_orientation(
                batch['crop_pos'].cpu().numpy(), batch['crop_frame'].cpu().numpy(), umbilicus,
                spec.ct_grid_scale/spec.grid_scale)).to(device)
        x = with_tube(model, batch)
        threshold = batch['hist'].new_full((), REFINE_ALL)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
            model.training_forward(x, batch['hist'], batch['hmask'], threshold)
    for h in hooks:
        h.remove()
    return {name: sums[name]/counts[name] for name in sums}


def transfer(source, cfg, means):
    """The state of a ``cfg`` model (single lattice) started from the two-level ``source`` model, and the names of its
    added parameters."""
    old_cfg = source.cfg
    if old_cfg.multiscale_fine_width is None or cfg.multiscale_fine_width is not None:
        raise ValueError('transfer goes from a two-level model to a single-lattice one')
    target = SheetStripFollower(cfg)
    state, old = target.state_dict(), {k: v.cpu() for k, v in source.state_dict().items()}
    fine, coarse = old_cfg.cnn_channels[-1], (old_cfg.multiscale_coarse_channels or old_cfg.cnn_channels)[-1]
    out, handled = {}, set()
    for name, value in state.items():
        if name.startswith('cnn.context_'):
            out[name] = value
        elif name.startswith('cnn.'):
            out[name] = widen(old[name], value) if old[name].shape != value.shape else old[name]
        else:
            continue
        handled.add(name)
    for name in TOKEN_PROJECTIONS:
        points = old[f'{name}.weight'].shape[1]//(fine+1)
        w, b = map_columns((old[f'{name}.weight'], old[f'{name}.bias']), (state[f'{name}.weight'], state[f'{name}.bias']),
                           points, fine, 0, True)
        out[f'{name}.weight'], out[f'{name}.bias'] = w, b
        handled |= {f'{name}.weight', f'{name}.bias'}
    n = cfg.strip_f_points
    for name, support in POINT_READERS.items():
        key = f'{name}.0'
        points = n if name == 'strip_refinement' else 1
        w, b = map_columns((old[f'{key}.weight'], old[f'{key}.bias']), (state[f'{key}.weight'], state[f'{key}.bias']),
                           points, fine, coarse, support, means.get(name))
        out[f'{key}.weight'], out[f'{key}.bias'] = w, b
        handled |= {f'{key}.weight', f'{key}.bias'}
    for name, value in state.items():
        if name in handled:
            continue
        if name not in old or old[name].shape != value.shape:
            raise ValueError(f'no source for {name} {tuple(value.shape)}')
        out[name] = old[name]
    dropped = sorted(k for k in old if k not in out)
    if any(not (k.startswith('coarse_cnn.') or k.startswith('cell_token.sector_projection_level')) for k in dropped):
        raise ValueError(f'unexpected source weights left over: {dropped}')
    return out, sorted(n for n, _ in target.named_parameters() if n.startswith('cnn.context_'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help="the source run's configuration (data for the input means)")
    ap.add_argument('--checkpoint', help="default: the source run's last.pt")
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--set', nargs='*', default=[], help='configuration KEY=JSON over the source model')
    ap.add_argument('--batches', type=int, default=8, help='training batches for the coarse columns\' mean inputs')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    run = RunConfig.load(args.config)
    path = args.checkpoint or str(Path(run.out_root)/run.name/'last.pt')
    source, old_cfg, ck = load_strip_model(path, args.device)
    values = {k: json.loads(v) for k, v in (o.split('=', 1) for o in args.set)}
    cfg = SheetStripConfig(**{**ck['model_cfg'], 'multiscale_fine_width': None, 'strip_full_fine': False, **values})
    spec = CTVolumeSpec.from_dict(ck['vol_spec'])
    loader, _ = build_data(run, old_cfg, spec, ck, 'train', augment=False, seed=4242, workers=4)
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == args.batches:
            break
    umbilicus = load_umbilicus(run.data.umbilicus) if old_cfg.strip_orientation else None
    means = mean_inputs(source, batches, umbilicus, spec, args.device)
    state, new_params = transfer(source, cfg, means)
    start = dict(ck, model_cfg=cfg.to_dict(), model=state, ema=state, step=0, best=None, new_params=new_params,
                 transfer=dict(source=str(Path(path).resolve()), mean_batches=len(batches), set=values))
    start.pop('optimizer', None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(start, args.out)
    model = SheetStripFollower(cfg)
    model.load_state_dict(state)
    print(f'{sum(p.numel() for p in source.parameters()):,} -> {sum(p.numel() for p in model.parameters()):,} '
          f'parameters; new: {new_params}; -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
