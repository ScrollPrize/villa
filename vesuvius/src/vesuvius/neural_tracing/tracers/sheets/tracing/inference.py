"""Trace a sheet from a seed polyline and a direction; write the traced rows as a tifxyz.

The seed is either a grid line of an existing tifxyz (``--seed-patch DIR --line row:40 --range 100:160
--direction +``: row 40, columns 100..160, advancing towards increasing rows) or a JSON file
``{"points": [[x, y, z], ...], "direction": [dx, dy, dz]}`` in level-0 voxels. The seed is re-sampled by arc
length at ``data.point_spacing``; the number of front points follows from its length (any count >= 3 works, the
model is independent of it). Each step reads the volume block around the current front. Rows are the steps,
columns the dense curve samples, ``trace.spline_samples`` per gap. Curve samples of gaps with a low-confidence front
point are written as invalid (-1); the points keep riding the curve, so they reappear when they become confident
again. Tracing stops after ``--steps``, at the first row with no confident point connected to the surface traced
so far, or when the front leaves the volume.

``--checkpoint``: a local file, or a Hugging Face model repo ``ORG/NAME[@REVISION]`` holding it (``shared/hub.py``:
``last.pt``, else its only ``.pt``); default ``DEFAULT_CHECKPOINT``. A checkpoint records the CT volume it was trained
on; ``--volume`` reads another (e.g. a local copy of that scan), ``--path-map`` rewrites recorded path prefixes.

  python -m vesuvius.neural_tracing.tracers.sheets.tracing.inference --volume /path/to/s1_ds2.zarr \\
      --seed-patch /path/to/patch --line row:40 --range 100:118 --direction + --steps 60 --threshold 0.3 --out traces/t0
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile
import torch

from vesuvius.neural_tracing.tracers.shared.geometry import resample_polyline
from vesuvius.neural_tracing.tracers.shared.hub import checkpoint_source, hf_repo, hub_folder
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import ChunkedArray
from vesuvius.neural_tracing.tracers.sheets import config as C
from vesuvius.neural_tracing.tracers.sheets.data.episodes import collate_fibers
from vesuvius.neural_tracing.tracers.sheets.checkpoint import load_checkpoint
from vesuvius.neural_tracing.tracers.sheets.tracing.rollout import Front, Tracer
from vesuvius.neural_tracing.tracers.sheets.sampling import view_radius
from vesuvius.tifxyz import Tifxyz, write_tifxyz


def resample(points: np.ndarray, spacing: float) -> np.ndarray:
    """A seed polyline's front points: ``shared.geometry.resample_polyline`` at the point spacing (exact spacing from
    the first point); a front needs at least 3 points."""
    seed = resample_polyline(np.asarray(points, np.float64), spacing)
    if len(seed) < 3:
        raise ValueError(f'A seed needs at least two point spacings ({2*spacing:g} voxels) of length')
    return seed


def seed_from_patch(path, line: str, span: str, direction: str):
    """Seed polyline and per-point advance direction from a tifxyz grid line."""
    axis, index = line.split(':')
    index = int(index)
    lo, hi = (int(v) for v in span.split(':'))
    xyz = np.stack([tifffile.imread(Path(path)/f'{c}.tif') for c in 'xyz'], -1).astype(np.float64)
    sign = 1 if direction == '+' else -1
    if axis == 'col':
        xyz = xyz.transpose(1, 0, 2)
    elif axis != 'row':
        raise ValueError("--line must be row:N or col:N")
    points, behind = xyz[index, lo:hi], xyz[index-sign, lo:hi]
    if np.any(points == -1) or np.any(behind == -1):
        raise ValueError('The seed line and the line behind it must be valid over --range')
    return points, points-behind


def seed_from_json(path):
    document = json.loads(Path(path).read_text())
    points = np.asarray(document['points'], np.float64)
    return points, np.broadcast_to(np.asarray(document['direction'], np.float64), points.shape)


def read_block(volume: ChunkedArray, pos: np.ndarray, pad: float, level: int):
    scale = 1 << level
    lo, hi = pos.min(0)-pad, pos.max(0)+pad
    origin = np.floor((lo[::-1]+.5)/scale-.5).astype(np.int64)
    size = np.ceil((hi-lo)[::-1]/scale).astype(np.int64)+2
    return volume.read(origin, size), origin


def _reach_row(kept: np.ndarray, above: np.ndarray) -> np.ndarray:
    """Kept columns of a row connected to the reached columns of the row above: each contiguous run of kept
    columns is reached if any of its columns is directly below a reached column."""
    reached = np.zeros_like(kept)
    start = None
    for c in range(len(kept)+1):
        if c < len(kept) and kept[c]:
            start = c if start is None else start
        elif start is not None:
            if above[start:c].any():
                reached[start:c] = True
            start = None
    return reached


def connected_keep(confs: np.ndarray, threshold: float) -> np.ndarray:
    """Points to write (T+1, C): confident and 4-connected through confident points to the seed row (row 0,
    always kept), so the output is one connected surface rather than separate islands."""
    keep = confs > threshold
    keep[0] = True
    labels, _ = ndimage.label(keep)
    return np.isin(labels, np.unique(labels[0]))


@torch.no_grad()
def trace(model, cfg, seed: np.ndarray, direction: np.ndarray, steps: int, device, stop_on_disconnect: bool = True,
          follow: bool = False, fibers: dict | None = None):
    """Rows (T+1, C, 3) including the seed and per-column confidences (T+1, C).

    C is the number of dense Hermite samples; each gets the lower confidence of its gap's two control points. With ``stop_on_disconnect`` (default), tracing stops at the first row with no
    confident point connected to the surface traced so far (e.g. an entirely missed row): later rows could only
    form separate components. That row is not returned. Use ``connected_keep`` for the points to write.
    ``follow``: fiber-following mode (a front across a fiber, advancing along it). ``fibers``: the fibers around the
    trace (``data.fibers.episode_fibers``, the seed fiber as instance 0), shown to a model with
    ``model.fiber_tokens``.
    """
    d = cfg.data
    count = len(seed)
    cfg = copy.deepcopy(cfg)
    cfg.data.front_points = count
    cfg.data.fiber_follow = follow  # fiber-following mode (model.fiber_follow_mode): a front across a fiber
    tracer = Tracer(model, cfg, device)
    unit = direction/np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    pos = torch.tensor(seed, dtype=torch.float32, device=device)[None]
    prev = torch.tensor(seed-d.step*unit, dtype=torch.float32, device=device)[None]
    front = Front(pos, prev, torch.ones(1, count, device=device))
    volume = ChunkedArray(f'{d.volume}/{d.volume_level}', 1 << 30)
    pad = view_radius(cfg.model)+d.step+4
    columns = lambda pos, conf, dense=None: tuple(x[0].cpu().numpy() for x in tracer.output_columns(pos, conf, dense))
    p, c = columns(front.pos, front.conf)
    rows, confs = [p], [c]
    reach = np.ones(len(c), bool)  # the seed row
    shape = np.asarray(volume.shape)[::-1]*(1 << d.volume_level)
    fiber_inputs = None
    if fibers is not None and cfg.model.fiber_tokens and len(fibers['points']):
        fiber_inputs = {k: v.to(device) for k, v in collate_fibers([dict(fibers=fibers)]).items()}
    model.eval()
    for _ in range(steps):
        block, origin = read_block(volume, front.pos[0].cpu().numpy(), pad, d.volume_level)
        block = torch.from_numpy(block).to(device).float()[None, None]
        out = tracer.step(front, block, torch.from_numpy(origin).to(device)[None], fiber_inputs)
        front = tracer.advance(front, out)
        p, c = columns(front.pos, front.conf, out.dense)
        reach = _reach_row(c > cfg.trace.confidence_threshold, reach)
        if stop_on_disconnect and not reach.any():
            break
        rows.append(p)
        confs.append(c)
        if np.any(p < 0) or np.any(p >= shape):
            break
    return np.stack(rows), np.stack(confs)


def surface_area_vx2(xyz: np.ndarray) -> float:
    """Area (voxels^2) of a tifxyz grid (R, C, 3), invalid nodes -1: each grid quad with four valid corners as the
    triangles (r, c)-(r, c+1)-(r+1, c+1) and (r, c)-(r+1, c+1)-(r+1, c), as VC3D's SurfaceAreaCalculator counts it."""
    p = np.asarray(xyz, np.float64)
    ok = np.all(p != -1, -1) & np.all(np.isfinite(p), -1)
    quad = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    tri = lambda u, v, w: .5*np.linalg.norm(np.cross(v-u, w-u), axis=-1)
    return float((tri(a, b, d)+tri(a, d, c))[quad].sum())


def add_area(meta_path: Path, xyz: np.ndarray, voxel_size_um: float | None) -> dict:
    """Write ``area_vx2`` and, with the level-0 voxel size (micrometres), ``area_cm2`` into a tifxyz meta.json (the
    keys VC3D reads)."""
    meta = json.loads(meta_path.read_text())
    meta['area_vx2'] = surface_area_vx2(xyz)
    if voxel_size_um:
        meta['area_cm2'] = meta['area_vx2']*voxel_size_um**2/1e8
        meta['voxel_size_um'] = voxel_size_um
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def write_surface(out: Path, rows: np.ndarray, keep: np.ndarray, row_step: float, column_step: float,
                  provenance: dict | None = None, overwrite: bool = False, voxel_size_um: float | None = None) -> None:
    """Rows (R, C, 3) as a tifxyz named after ``out`` (stable uuid); points outside ``keep`` are written as -1.
    meta.json gets the area of the kept surface (``add_area``; cm^2 with ``voxel_size_um``)."""
    xyz = np.where(keep[..., None], rows, -1.).astype(np.float32)
    surface = Tifxyz(_x=xyz[..., 0], _y=xyz[..., 1], _z=xyz[..., 2], uuid=out.name,
                     _scale=(1./row_step, 1./column_step))
    write_tifxyz(out, surface, overwrite=overwrite)
    add_area(out/'meta.json', xyz, voxel_size_um)
    if provenance is not None:
        (out/'trace.json').write_text(json.dumps(provenance, indent=2, default=str))


def write_trace(out: Path, rows: np.ndarray, confs: np.ndarray, cfg, threshold: float, provenance: dict,
                overwrite: bool = False):
    keep = connected_keep(confs, threshold)
    column = cfg.data.point_spacing/cfg.trace.spline_samples
    write_surface(out, rows, keep, cfg.data.step, column, provenance, overwrite=overwrite,
                  voxel_size_um=cfg.data.voxel_size_um)
    np.save(out/'confidence.npy', confs.astype(np.float32))


def two_way_trace(up: tuple, down: tuple) -> tuple:
    """Two traces from the same seed row in opposite directions as one surface: ``up``/``down`` are (rows (T+1, C, 3),
    keep (T+1, C), confs (T+1, C)), each with the seed as row 0. Returns rows, keep and confs of the ``down`` rows
    reversed, the seed row (``up``'s), then the ``up`` rows; the seed row's index is ``len(down rows) - 1``."""
    return tuple(np.concatenate([d[:0:-1], u]) for u, d in zip(up, down))


def write_two_way_trace(out: Path, up: tuple, down: tuple, cfg, threshold: float, provenance: dict,
                        overwrite: bool = False):
    """``up`` and ``down`` (rows, confs) traced from the same seed row in opposite directions, written as one tifxyz
    (``two_way_trace``); each half keeps the points connected to the seed (``connected_keep``)."""
    halves = [(rows, connected_keep(confs, threshold), confs) for rows, confs in (up, down)]
    rows, keep, confs = two_way_trace(*halves)
    column = cfg.data.point_spacing/cfg.trace.spline_samples
    write_surface(out, rows, keep, cfg.data.step, column, dict(provenance, seed_row=len(down[0])-1),
                  overwrite=overwrite, voxel_size_um=cfg.data.voxel_size_um)
    np.save(out/'confidence.npy', confs.astype(np.float32))


# The 7-point sheet tracer (remote_l0_neighbor: 7-point fronts, points 60 vx apart; best near --threshold 0.3).
DEFAULT_CHECKPOINT = 'scrollprize/sheet_tracer_7pt_gap60'
CHECKPOINT_NAME = 'last.pt'


def resolve_checkpoint(checkpoint) -> str:
    """The local file of ``checkpoint``: a file, or a Hugging Face repo (``shared/hub.py``) holding ``CHECKPOINT_NAME``
    or exactly one ``.pt``."""
    repo = hf_repo(checkpoint)
    if repo is None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise SystemExit(f'Checkpoint not found: {checkpoint}')
        return str(path)
    folder = hub_folder(repo)
    if (folder/CHECKPOINT_NAME).is_file():
        return str(folder/CHECKPOINT_NAME)
    found = sorted(folder.glob('*.pt'))
    if len(found) != 1:
        raise SystemExit(f'Hugging Face repo {repo[0]} needs {CHECKPOINT_NAME} or exactly one .pt; it has '
                         f'{[f.name for f in found]}')
    return str(found[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                    help=f'a checkpoint file, or a Hugging Face repo ORG/NAME[@REVISION] holding it (default: '
                         f'{DEFAULT_CHECKPOINT})')
    ap.add_argument('--volume', help="CT volume (OME-Zarr pyramid) to trace in, read at the checkpoint's level "
                                     "(default: the checkpoint's)")
    ap.add_argument('--seed-patch')
    ap.add_argument('--line', help='row:N or col:N of --seed-patch')
    ap.add_argument('--range', help='lo:hi along the seed line')
    ap.add_argument('--direction', choices=('+', '-'), default='+')
    ap.add_argument('--seed-json')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--threshold', type=float, help="confidence threshold (default: the checkpoint's)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW',
                    help='replace the path prefix OLD by NEW in the checkpoint\'s data paths (volumes, patch '
                         'directories), e.g. for a checkpoint trained on another machine; repeatable')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--voxel-size', type=float, help="level-0 voxel size (um) of --volume for area_cm2 (default: "
                                                     "its meta.json voxelsize, else the checkpoint's)")
    args = ap.parse_args(argv)
    path = resolve_checkpoint(args.checkpoint)
    model, cfg, _ = load_checkpoint(path, args.device)
    C.path_mapper(args.path_map)(cfg)
    if args.volume:
        cfg.data.volume = args.volume
        meta = Path(args.volume)/'meta.json'  # volpkg volume metadata, as neural_tracing/infer.py reads it
        voxel_size = args.voxel_size or (json.loads(meta.read_text()).get('voxelsize') if meta.is_file() else None)
        if not voxel_size:
            print(f'No voxel size for {args.volume} (meta.json voxelsize or --voxel-size): area_cm2 assumes '
                  f'{cfg.data.voxel_size_um} um')
        cfg.data.voxel_size_um = voxel_size or cfg.data.voxel_size_um
    elif args.voxel_size:
        cfg.data.voxel_size_um = args.voxel_size
    if args.threshold is not None:
        cfg.trace.confidence_threshold = args.threshold
    if args.seed_json:
        points, direction = seed_from_json(args.seed_json)
    else:
        points, direction = seed_from_patch(args.seed_patch, args.line, args.range, args.direction)
    seed = resample(points, cfg.data.point_spacing)
    direction = np.stack([np.interp(np.linspace(0, len(points)-1, len(seed)), np.arange(len(points)), direction[:, i])
                          for i in range(3)], 1)
    rows, confs = trace(model, cfg, seed, direction, args.steps, args.device)
    out = Path(args.out)
    write_trace(out, rows, confs, cfg, cfg.trace.confidence_threshold,
                dict(vars(args), checkpoint=checkpoint_source(args.checkpoint, path), volume=cfg.data.volume,
                     front_points=len(seed), steps_traced=len(rows)-1))
    kept = connected_keep(confs, cfg.trace.confidence_threshold)[1:].mean() if len(rows) > 1 else 0.
    print(f'{len(rows)-1} steps x {len(seed)} front points ({rows.shape[1]} columns), {kept:.0%} kept -> {out}')


if __name__ == '__main__':
    main()
