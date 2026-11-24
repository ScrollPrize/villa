"""Sheet-strip decisions batched across fibers: CPU workers build each decision's geometry, one process samples the
crops on the GPU and runs both models over many fibers at once.

``infer.StripPredictor.predict`` builds a fiber's decisions a few at a time in one process: CPU crops, the frame model on
the CPU, then the strip model. Here the same decisions are split around the frame model:

- ``StripGeometry.plan`` (worker): the decisions as ``predict`` builds them, without training labels
  (``decision_on_path(labels=False)``) or identity evidence, which are never model inputs, and with the seed heading
  resolved as ``predict`` does; returns the frame model's inputs (``FrameRequest``).
- ``StripEngine.frame_outputs`` (GPU process): the frame model over the requests of many fibers, on the GPU with crops
  from vc_reader's CUDA or Metal reader (``crops='gpu'``), else on the CPU with CPU crops, as ``predict`` runs it.
- ``StripGeometry.finish`` (worker): the learned frame, the crop axis of the previous prediction, the references,
  path features, tube points and crop orientation (``StripInputs``), as the observation builder sets them.
- ``StripEngine.heights`` (GPU process): the strip crops on the GPU and the strip model, by default the decisions of
  each fiber in ``predict``'s batches of ``batch`` (per-sample results do not depend on which decisions share a batch
  of one size; a fiber's last, smaller batch keeps its size); with ``across_fibers``, consecutive batches of
  ``batch`` decisions from any fibers (faster at a larger batch; bf16 results then move by the batch size's rounding).
- ``StripGeometry.strip`` (worker): orientation, stitching and trimming as ``infer_all.strip_of_fiber``.

``run`` drives the workers and the engine; a ``FiberJob`` loads each fiber and writes its strip.
"""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
import queue
import time
from typing import Any

import numpy as np


@dataclass
class FrameRequest:
    """The frame model's inputs for a fiber's decisions: head positions (K, 3), prior crop frames (K, 3, 3; +z along
    the prior heading, heading_model.prior_frames), path features (K, F) and H/V families (K,) as 0/1."""
    positions: np.ndarray
    frames: np.ndarray
    paths: np.ndarray
    families: np.ndarray


@dataclass
class StripInputs:
    """A fiber's strip-model inputs without the crop: per decision its final position (K, 3) and frame (K, 3, 3),
    the model's ``x`` entries other than ``fine``, ``hist``/``hmask``, the tube points (ragged, crop-local) and the crop
    orientation (K, 5; None without ``strip_orientation``)."""
    positions: np.ndarray
    frames: np.ndarray
    x: dict
    hist: np.ndarray
    hmask: np.ndarray
    tube: list
    orientation: Any


class StripGeometry:
    """The CPU half of ``StripPredictor.predict`` for one process: everything but the crops and the models."""

    def __init__(self, cfg, sample_cfg, crop_reach, umbilicus, trace_per_patch, ct, spec, frame_window):
        self.cfg, self.sample_cfg, self.crop_reach = cfg, sample_cfg, crop_reach
        self.umbilicus, self.trace_per_patch, self.ct, self.spec = umbilicus, trace_per_patch, ct, spec
        self.frame_window = frame_window  # the frame model's path window (HeadingConfig.window)

    @classmethod
    def of(cls, predictor):
        """The geometry of a StripPredictor's checkpoint, CT and umbilicus. The batched path batches the frame
        model's requests, so it needs a checkpoint with one."""
        frames = predictor.builder.frames()
        if frames is None:
            raise ValueError('Batched inference needs a checkpoint with a frame model')
        p = predictor
        return cls(p.cfg, p.sample_cfg, p.crop_reach, p.umbilicus, p.trace_per_patch, p.ct, p.spec,
                   frames.model.cfg.window)

    def plan(self, fiber, arcs):
        """The decisions at ``arcs`` before their learned frames, and the frame model's request. Raises
        SeedHeadingError for a fiber whose seed has no usable CT orientation, as ``predict`` does."""
        from vesuvius.neural_tracing.tracers.fibers.data.targets import resolve_trace_seed
        from vesuvius.neural_tracing.tracers.fibers.frames.ct import fiber_family
        from vesuvius.neural_tracing.tracers.fibers.heading_model.model import path_features, prior_frames
        from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer import (
            decision_item, reuse_seed_heading, skip_remote_seed_heading)
        from vesuvius.neural_tracing.tracers.shared.reference import observed_path
        items, seeds = [], {}
        for t in arcs:
            item = decision_item(fiber, float(t), self.sample_cfg, self.cfg.n_history,
                                 self.sample_cfg.prediction_window, labels=False)
            # IdentityObservationBuilder.prepare without augmentation: the family; its identity curve and evidence
            # are diagnostics, and the references it lays out are laid out again once the frame is final
            item['fiber_family'] = fiber.tag
            skip_remote_seed_heading(item, self.crop_reach)
            reuse_seed_heading(item, self.ct, self.spec, seeds)
            resolve_trace_seed(item, self.ct, self.spec)
            if 'frame_policy' in item or not item.get('fiber_family'):
                raise ValueError('Batched strips need learned frames for every decision')
            items.append(item)
        priors = [item['frame'][:, 2] for item in items]
        frames = prior_frames(priors)
        paths = np.stack([path_features(observed_path(item), item['pos'], frame, self.frame_window)
                          for item, frame in zip(items, frames)])
        families = np.array([0 if fiber_family(item['fiber_family']) == 'H' else 1 for item in items])
        return items, FrameRequest(np.stack([np.asarray(item['pos'], np.float64) for item in items]),
                                   np.stack(frames), paths, families)

    def finish(self, fiber, items, request, heading, normal):
        """The strip-model inputs of ``fiber``'s planned ``items`` given the frame model's outputs
        (``heading``/``normal`` (K, 3) float32, crop-local) for ``request``."""
        import torch
        from vesuvius.neural_tracing.tracers.fibers.data.observations import reference_layout
        from vesuvius.neural_tracing.tracers.fibers.frames.ct import LEARNED_FRAME_POLICY, reframe_item
        from vesuvius.neural_tracing.tracers.fibers.frames.learned import apply_prediction_axis
        from vesuvius.neural_tracing.tracers.fibers.heading_model.model import HeadingPredictor
        from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import crop_orientation, polyline_points
        from vesuvius.neural_tracing.tracers.models.path_geometry import path_geometry_inputs
        priors = [item['frame'][:, 2] for item in items]
        out = dict(heading=torch.from_numpy(heading), normal=torch.from_numpy(normal))
        heads, normals = HeadingPredictor._world_normals(list(request.frames), out, priors)
        frames = HeadingPredictor._world_frames(heads, normals, None)
        cfg, crop = self.cfg, self.cfg.fine
        for item, frame in zip(items, frames):
            # frames/learned._orient_items, then orient_items and the builder's finalize_frames
            reframe_item(item, frame)
            item.update(frame_policy=LEARNED_FRAME_POLICY, ct_frame_diagnostics=dict(source=3, energy=0., gap=0.))
            apply_prediction_axis(item, self.sample_cfg.prediction_window)
            item.pop('prev_prediction', None)
            reference_layout(item, cfg)
        stack = lambda key: np.stack([item[key] for item in items]).astype(np.float32)
        x = dict(seed=stack('visible_seed'), seed_mask=stack('visible_seed_mask'), seed_age=stack('visible_seed_age'),
                 seed_tangent=stack('visible_seed_tangent'))
        x.update({k: v.numpy() for k, v in path_geometry_inputs(items).items()})
        quality = [item.get('ct_frame_diagnostics', {}) for item in items]
        for key, default, dtype in (('source', -1, np.int64), ('energy', 0., np.float32), ('gap', 0., np.float32)):
            x['ct_frame_'+key] = np.array([q.get(key, default) for q in quality], dtype=dtype)
        positions = np.stack([np.asarray(item['pos'], np.float64) for item in items])
        frames = np.stack([np.asarray(item['frame'], np.float64) for item in items])
        tube = [polyline_points(fiber.points, positions[k], frames[k], crop, crop.spacing/2,
                                3*cfg.tube_sigma*crop.spacing, TUBE_POINTS) for k in range(len(items))]
        orientation = None
        if self.umbilicus is not None:
            orientation = crop_orientation(positions, frames, self.umbilicus, self.trace_per_patch)
        return StripInputs(positions, frames, x, np.stack([i['hist_local'] for i in items]).astype(np.float32),
                           np.stack([i['hmask'] for i in items]).astype(np.float32), tube, orientation)


TUBE_POINTS = 2048  # infer.StripPredictor.predict's tube buffer


class StripEngine:
    """The model half of ``StripPredictor.predict`` over the decisions of many fibers per call: the strip model on
    ``predictor``'s device, and the frame model on its own predictor's (as ``predict`` runs it: the GPU with
    ``crops='gpu'``, else the CPU). Crops come from vc_reader's CUDA or Metal reader with ``crops='gpu'``, else from the
    CPU sampler (on the CPU, the engine then reproduces ``predict``)."""

    def __init__(self, predictor, batch=4, frame_chunk=64, across_fibers=False, compile_mode=None):
        """``compile_mode``: a torch.compile mode for the strip model after the tube render (with ``across_fibers``:
        every batch is then padded to ``batch`` decisions so the compiled graph keeps one shape). Compiling turns on
        inductor's ``emulate_precision_casts`` for the whole process."""
        import torch
        if torch.device(predictor.device).type == 'mps':
            # the engine's launches have neither predict's MPS batch padding nor its queue flushes; on MPS one process
            # fiber by fiber with across-fiber batches (StripPredictor.predict_many) keeps the GPU busy
            raise ValueError('Batched mode (--workers) runs on CUDA or the CPU; on MPS run without --workers, with '
                             '--across-fibers')
        self.predictor, self.batch, self.frame_chunk = predictor, batch, frame_chunk
        self.across_fibers = across_fibers
        if compile_mode is not None and not across_fibers:
            raise ValueError('A compiled strip model needs across_fibers (one batch shape)')
        if compile_mode is not None:
            import torch._inductor.config
            # fused kernels otherwise skip eager's intermediate bf16 roundings: plain compile made strips measurably
            # worse than eager against stock on PHerc0211 fibres, while emulated casts stayed within noise. A
            # global, since torch.compile takes a mode or inductor options, not both
            torch._inductor.config.emulate_precision_casts = True
        model = predictor.model

        def strip_model(x, hist, hmask):
            ctx = model.context(x, hist, hmask)
            out, scale = model.strip_outputs(ctx)
            return out[:, -1].float(), None if scale is None else scale.float()
        self.strip_model = strip_model if compile_mode is None else torch.compile(strip_model, mode=compile_mode)
        self.device = torch.device(predictor.device)
        frames = predictor.builder.frames()
        self.frame_model, self.frame_device = frames.model, frames.device

    def crops(self, items, crop, device=None):
        """(B, 1, D, H, W) z-scored float32 crops of ``items``: from the predictor's GPU crop backend with
        ``crops='gpu'`` (vc_reader's CUDA or Metal reader, on the engine's device), else from the CPU sampler, moved to
        ``device`` (default: the engine's)."""
        from vesuvius.neural_tracing.tracers.shared.crop_sampling.crops import scalar_crops
        p = self.predictor
        if p.gpu_crops is not None:
            return p.gpu_crops(items, p.ct, p.spec, crop)
        return scalar_crops(items, p.ct, p.spec, crop).to(device or self.device)

    def frame_outputs(self, requests):
        """Frame-model outputs (heading, normal; each (K, 3) float32 per request, crop-local) of ``requests``, in
        chunks of ``frame_chunk`` decisions as HeadingPredictor.predict_frames_batched runs them: each request's
        decisions in chunks of their own, as ``predict`` batches them (``frame_chunk`` = its ``batch``), or with
        ``across_fibers`` consecutive chunks of all of them."""
        import torch
        positions = np.concatenate([r.positions for r in requests])
        frames = np.concatenate([r.frames for r in requests])
        paths = torch.from_numpy(np.concatenate([r.paths for r in requests]))
        families = torch.from_numpy(np.concatenate([r.families for r in requests]))
        bounds = np.cumsum([0]+[len(r.positions) for r in requests])
        # a decision's frame can change in the last bits with the batch it runs in, so only predict's batches
        # reproduce predict exactly
        spans = [(0, len(positions))] if self.across_fibers else list(zip(bounds[:-1], bounds[1:]))
        chunk = self.frame_chunk
        parts = [slice(a, min(a+chunk, hi)) for lo, hi in spans for a in range(lo, hi, chunk)]
        crop = self.frame_model.cfg.patch
        heading, normal = [], []
        with torch.no_grad():
            for part in parts:
                items = [dict(pos=a, frame=b) for a, b in zip(positions[part], frames[part])]
                # with CPU crops the frame model stays on the CPU, as in predict, while the strip model runs on the GPU
                patch = self.crops(items, crop, self.frame_device)
                out = self.frame_model.forward_outputs(patch, paths[part].to(self.frame_device),
                                                       families[part].to(self.frame_device))
                heading.append(out['heading'])
                normal.append(out['normal'])
        heading, normal = (torch.cat(v).float().cpu().numpy() for v in (heading, normal))
        return [(heading[a:b], normal[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]

    def heights(self, inputs, crop_chunk=32):
        """Strip heights (K, Nf, Nv) and log error scales (K, Nf, Nv; None without ``strip_uncertainty``) of each
        fiber's ``StripInputs``. Crops are sampled ``crop_chunk`` decisions at a time (each crop's values do not depend
        on the others); the strip model runs each fiber's decisions in ``predict``'s batches (``batch``, then the
        rest), so every decision sees a batch of the size stock gives it, or with ``across_fibers`` all decisions in
        consecutive batches of ``batch``."""
        import torch
        from vesuvius.neural_tracing.tracers.fibers.sheet_strip.train import with_tube
        from vesuvius.neural_tracing.tracers.shared.tensors import move_batch
        p, crop = self.predictor, self.predictor.cfg.fine
        positions = np.concatenate([f.positions for f in inputs])
        frames = np.concatenate([f.frames for f in inputs])
        x = {key: np.concatenate([f.x[key] for f in inputs]) for key in inputs[0].x}
        hist, hmask = (np.concatenate([getattr(f, k) for f in inputs]) for k in ('hist', 'hmask'))
        tubes = [curve for f in inputs for curve in f.tube]
        orientation = None if inputs[0].orientation is None else np.concatenate([f.orientation for f in inputs])
        groups, start = [], 0
        for f in inputs:
            groups += [(start+a, start+min(a+self.batch, len(f.positions))) for a in range(0, len(f.positions), self.batch)]
            start += len(f.positions)
        if self.across_fibers:
            groups = [(a, min(a+self.batch, start)) for a in range(0, start, self.batch)]
            crop_chunk = self.batch  # one crop call per batch
        heights = torch.empty((start, len(p.f), len(p.v)), dtype=torch.float32, device=self.device)
        scales = None
        with torch.no_grad():
            for c0 in range(0, start, crop_chunk):
                c1 = min(c0+crop_chunk, start)
                items = [dict(pos=a, frame=b) for a, b in zip(positions[c0:c1], frames[c0:c1])]
                fine = self.crops(items, crop)
                for a, b in groups:
                    if not c0 <= a < c1:
                        continue
                    if b > c1:  # a batch across two crop chunks: its own crops
                        part_items = [dict(pos=u, frame=w) for u, w in zip(positions[a:b], frames[a:b])]
                        part_fine = self.crops(part_items, crop)
                    else:
                        part_fine = fine[a-c0:b-c0]
                    tube = np.zeros((b-a, TUBE_POINTS, 3), np.float32)
                    valid = np.zeros((b-a, TUBE_POINTS), bool)
                    for row, curve in enumerate(tubes[a:b]):
                        tube[row, :len(curve)], valid[row, :len(curve)] = curve, True
                    batch = dict(x={key: torch.from_numpy(value[a:b]) for key, value in x.items()},
                                 hist=torch.from_numpy(hist[a:b]), hmask=torch.from_numpy(hmask[a:b]),
                                 tube_points=torch.from_numpy(tube), tube_valid=torch.from_numpy(valid))
                    if orientation is not None:
                        batch['crop_orientation'] = torch.from_numpy(orientation[a:b])
                    device_batch = move_batch(batch, p.device)
                    device_batch['x']['fine'] = part_fine
                    if self.across_fibers and b-a < self.batch:  # pad to the batch's one shape; padding is discarded
                        device_batch = _padded(device_batch, self.batch)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                        out, scale = self.strip_model(with_tube(p.model, device_batch), device_batch['hist'],
                                                      device_batch['hmask'])
                    heights[a:b] = out[:b-a]
                    if scale is not None:
                        if scales is None:
                            scales = torch.empty_like(heights)
                        scales[a:b] = scale[:b-a]
        heights = heights.cpu().numpy().astype(np.float64)
        scales = None if scales is None else scales.cpu().numpy()
        bounds = np.cumsum([0]+[len(f.positions) for f in inputs])
        return [(heights[a:b], None if scales is None else scales[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]


def _padded(batch, size):
    """``batch`` (nested dicts of tensors, decisions first) with its last decision repeated to ``size`` rows."""
    import torch
    if isinstance(batch, dict):
        return {k: _padded(v, size) for k, v in batch.items()}
    return torch.cat((batch, batch[-1:].expand(size-len(batch), *batch.shape[1:])))


class FiberJob:
    """What ``run`` does per fiber, in the workers: ``geometry()`` builds the worker's StripGeometry once, ``load(key)``
    returns (fiber, arcs) or None to skip it, ``write(key, fiber, arcs, strip, raw)`` saves the stitched strip (rows,
    kept nodes, expected error) and returns its index record (``raw``: the decisions' ``DecisionOutputs``), and
    ``failed(key, error)`` the record of a fiber whose seed heading failed. Must be picklable."""

    def geometry(self):
        raise NotImplementedError

    def load(self, key):
        raise NotImplementedError

    def write(self, key, fiber, arcs, strip, raw):
        raise NotImplementedError

    def failed(self, key, error):
        raise NotImplementedError


@dataclass
class DecisionOutputs:
    """A fiber's per-decision results, as ``StripPredictor.predict`` returns them: heights (K, Nf, Nv), log error
    scales (K, Nf, Nv) or None, final frames (K, 3, 3) and positions (K, 3)."""
    heights: np.ndarray
    scales: Any
    frames: np.ndarray
    positions: np.ndarray


def stitched_strip(predictor_grid, fiber, arcs, heights, scales, frames, positions, blend, trim):
    """``infer_all.strip_of_fiber`` after ``predict``: rows (R, Nv, 3) trace voxels, kept nodes, expected |error| per
    node or None. ``predictor_grid``: (f, v, trace_per_patch); ``trim`` in trace voxels."""
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer import orient_decisions, stitch, stitch_values, trim_ends
    f, v, tpp = predictor_grid
    oriented, oriented_frames, flipped = orient_decisions(heights, frames)
    rows, weight, _ = stitch(oriented, oriented_frames, positions, arcs, f, v, blend)
    keep = trim_ends(weight > 0, arcs[0]+f[0], f[1]-f[0], fiber.length, trim)
    expected = None
    if scales is not None:
        error = np.exp(np.where(flipped[:, None, None], scales[:, :, ::-1], scales))/tpp
        expected = stitch_values(error, arcs, f, blend)
    return rows, keep, expected


def _worker(job, inbox, outbox):
    """A worker process: plans fibers, finishes their frames, stitches and writes their strips."""
    import torch
    from vesuvius.neural_tracing.tracers.shared.ct_structure import SeedHeadingError
    torch.set_num_threads(1)
    geometry = job.geometry()
    state = {}
    while True:
        message = inbox.get()
        kind, key = message[0], message[1]
        if kind == 'stop':
            return
        try:
            if kind == 'fiber':
                loaded = job.load(key)
                if loaded is None:
                    outbox.put(('skipped', key, None))
                    continue
                fiber, arcs = loaded
                try:
                    items, request = geometry.plan(fiber, arcs)
                except SeedHeadingError as error:
                    outbox.put(('done', key, job.failed(key, error)))
                    continue
                state[key] = (fiber, arcs, items, request)
                outbox.put(('frames', key, request))
            elif kind == 'frames':
                fiber, arcs, items, request = state[key]
                inputs = geometry.finish(fiber, items, request, *message[2])
                state[key] = (fiber, arcs, inputs)
                outbox.put(('strip', key, inputs))
            elif kind == 'heights':
                fiber, arcs, inputs = state.pop(key)
                raw = DecisionOutputs(*message[2], inputs.frames, inputs.positions)
                strip = stitched_strip(message[3], fiber, arcs, raw.heights, raw.scales, raw.frames, raw.positions,
                                       *message[4])
                outbox.put(('done', key, job.write(key, fiber, arcs, strip, raw)))
        except Exception as error:  # the run stops on any other failure, naming the fiber
            import traceback
            outbox.put(('error', key, f'{type(error).__name__}: {error}\n{traceback.format_exc()}'))
            return


def run(engine, job, keys, *, workers=8, in_flight=3, frame_batch=256, strip_batch=128, blend=16., trim=0.,
        on_record=None, log=None):
    """Strips of the fibers ``keys`` (``job`` loads and writes them) over ``workers`` CPU processes and ``engine``
    (StripEngine) in this one. Each worker holds up to ``in_flight`` fibers; frame-model and strip-model work runs once
    ``frame_batch``/``strip_batch`` decisions wait, or sooner when nothing else is ready. ``on_record`` sees each
    fiber's index record (None for a skipped fiber). Returns the decisions run."""
    p = engine.predictor
    grid = (p.f, p.v, p.trace_per_patch)
    context = multiprocessing.get_context('spawn')  # the engine's process holds CUDA: no fork
    outbox = context.Queue()
    inboxes = [context.Queue() for _ in range(workers)]
    procs = [context.Process(target=_worker, args=(job, inboxes[w], outbox), daemon=True) for w in range(workers)]
    for proc in procs:
        proc.start()
    keys = list(keys)
    owner, load, next_key = {}, [0]*workers, 0
    frames_waiting, strips_waiting = [], []
    done = decisions = 0
    started = time.time()

    def feed():
        nonlocal next_key
        for w in range(workers):
            while load[w] < in_flight and next_key < len(keys):
                key = keys[next_key]
                next_key += 1
                owner[key] = w
                load[w] += 1
                inboxes[w].put(('fiber', key))

    def take(message):
        nonlocal done, decisions
        kind, key, payload = message
        if kind == 'error':
            raise RuntimeError(f'fiber {key}: {payload}')
        if kind == 'frames':
            frames_waiting.append((key, payload))
        elif kind == 'strip':
            strips_waiting.append((key, payload))
        else:
            load[owner.pop(key)] -= 1
            done += 1
            if payload is not None:
                decisions += int(payload.get('decisions', 0))
            if on_record is not None:
                on_record(key, payload)

    try:
        feed()
        while done < len(keys):
            try:
                take(outbox.get(timeout=.05 if (frames_waiting or strips_waiting) else 5.))
                while True:
                    take(outbox.get_nowait())
            except queue.Empty:
                pass
            for proc in procs:
                if not proc.is_alive() and proc.exitcode not in (0, None):
                    raise RuntimeError(f'worker {proc.pid} died with exit code {proc.exitcode}')
            # every fiber in flight waits on the engine: nothing else will arrive, so both kinds of work run now
            # (otherwise frames waiting for strips to clear and strips waiting for frames to clear never run)
            stalled = len(frames_waiting)+len(strips_waiting) >= sum(load)
            idle = not frames_waiting or not strips_waiting or stalled
            if frames_waiting and (sum(len(r.positions) for _, r in frames_waiting) >= frame_batch or idle):
                outputs = engine.frame_outputs([r for _, r in frames_waiting])
                for (key, _), out in zip(frames_waiting, outputs):
                    inboxes[owner[key]].put(('frames', key, out))
                frames_waiting.clear()
            if strips_waiting and (sum(len(s.positions) for _, s in strips_waiting) >= strip_batch or
                                   not frames_waiting or stalled):
                outputs = engine.heights([s for _, s in strips_waiting])
                for (key, _), out in zip(strips_waiting, outputs):
                    inboxes[owner[key]].put(('heights', key, out, grid, (blend, trim)))
                strips_waiting.clear()
            feed()
            if log is not None:
                log(done, len(keys), decisions, time.time()-started)
    finally:
        for inbox in inboxes:
            inbox.put(('stop', None))
        for proc in procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
    return decisions
