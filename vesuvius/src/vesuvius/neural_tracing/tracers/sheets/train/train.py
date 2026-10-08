"""Train the sheet tracer on rollouts over verified patches.

Each iteration seeds a front from GT rows (with noise), rolls it out ``data.steps`` steps with the model's own
predictions (noise between steps; in snapped episodes front_points beyond ``train.reset_distance`` are snapped back
onto the sheet, free episodes drift; some episodes push a run of front_points towards a neighbouring sheet), and
averages the per-step losses. The run directory ``output/NAME`` holds config.json, log.jsonl, last.pt,
best.pt (highest validation score = coverage - kept_off), periodic diagnostics (val_XXXXXX/), and a final long-rollout eval.

Usage:
  python -m vesuvius.neural_tracing.tracers.sheets.train.train --config remote_l0_points_ft.json --init output/l0_joint/last.pt
  python -m vesuvius.neural_tracing.tracers.sheets.train.train --config remote_l0_fiber_ft.json \
      --init output/remote_l0_points_ft/last.pt
  python -m vesuvius.neural_tracing.tracers.sheets.train.train --resume output/remote_l0_fiber_ft/last.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import queue
import signal
import sys
import threading
import time

import torch
from tqdm import tqdm

from vesuvius.neural_tracing.tracers.shared.system import raise_open_file_limit
from vesuvius.neural_tracing.tracers.sheets import config as C
from vesuvius.neural_tracing.tracers.sheets.checkpoint import load_checkpoint
from vesuvius.neural_tracing.tracers.sheets.evaluation import rollouts as E
from vesuvius.neural_tracing.tracers.sheets.data.episodes import (
    FIBER_KEYS, EpisodeSampler, EpisodeStream, MixedFrontStream, collate, episode_pad, fixed_episodes, keep_batch)
from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index
from vesuvius.neural_tracing.tracers.sheets.train.live import LiveChains
from vesuvius.neural_tracing.tracers.sheets.losses import step_losses
from vesuvius.neural_tracing.tracers.models.blocks import LOGIT_SCALE_LR_SCALE
from vesuvius.neural_tracing.tracers.sheets.model import build_model, compile_model, init_model_state, legacy_state, load_model_state
from vesuvius.neural_tracing.tracers.sheets.tracing.rollout import Tracer
from vesuvius.neural_tracing.tracers.sheets.tracing.seeding import perturb, seed_rows, sheet_jump
from vesuvius.neural_tracing.tracers.sheets.evaluation.figures import episode_figure, metric_figure, training_figure
from vesuvius.neural_tracing.tracers.shared.distributed import from_environment
from vesuvius.neural_tracing.tracers.shared.io import atomic_torch_save
from vesuvius.neural_tracing.tracers.shared.training import clip_training_gradients, lr_at, prepare_run_dir, update_ema


def learning_rate(it: int, tc, minutes: float = 0.) -> float:
    """The tracers' warmup-cosine schedule (``shared.training.lr_at``) at iteration ``it`` (from 0), decaying to zero at
    whichever comes first: ``iterations`` or ``max_minutes`` of training (``minutes`` so far, including resumed time)."""
    return lr_at(it+1, tc.lr, tc.warmup, tc.iterations, minutes/tc.max_minutes if tc.max_minutes else None)


def train_rollout(tracer: Tracer, b: dict, cfg, spec, generator, front=None):
    """Loss averaged over the rollout steps, per-step stats averaged likewise, the final front, and the point
    confidences (B, N) of every step. ``front``: continue these fronts (live chains) instead of seeding from GT.
    Steps are detached from each other, and each step's loss is backpropagated right after the step (the same
    gradient as the summed loss, with only one step's activations in memory); the returned loss is detached.

    A ``free_fraction`` of episodes are never snapped back; in ``jump_probability`` of episodes a run of front_points
    is pushed onto a neighbouring sheet before one random step (not the last). With ``train.history_rows`` (``cfg``
    then has that many more ``data.history`` rows), the front first traces 0..history_rows GT rows up to the seed
    without loss, put back on GT after each step, to fill the grid memory.
    """
    tc = cfg.train
    B, steps, device = b['gt'].shape[0], cfg.data.steps, b['gt'].device
    if front is None:
        seeds = seed_rows(b['gt'], b['valid'], cfg, spec, generator, noise=True)
        if 'seed_mask' in b:  # fiber-seeded episodes start from their own seed rows
            seeds = torch.where(b['seed_mask'][:, None, None, None], b['seed'].to(seeds.dtype), seeds)
        warm = int(torch.randint(0, tc.history_rows+1, (1,), device=device, generator=generator)) if tc.history_rows else 0
        H = seeds.shape[1]
        front = tracer.initial(seeds[:, :H-warm])
        with torch.no_grad():
            for j in range(warm):
                out = tracer.step(front, b['block'], b['origin'])
                front = tracer.teacher_forced(tracer.advance(front, out), seeds[:, H-warm+j])
    snap = torch.rand(B, device=device, generator=generator) >= tc.free_fraction
    jump = torch.rand(B, device=device, generator=generator) < tc.jump_probability
    jump_at = torch.randint(0, max(1, steps-1), (B,), device=device, generator=generator)
    total, stats, resets, jumped, confs = 0., {}, 0., 0., []
    fibers = {k: b[k] for k in FIBER_KEYS if k in b} or None  # fiber-started episodes: the fibers around them
    for t in range(steps):
        if tc.jump_probability > 0:
            pos, run = sheet_jump(front.pos, b['gt'], b['valid'], cfg, jump & (jump_at == t), generator,
                                  b['block'], b['origin'])
            tracer.move(front, pos)
            jumped += run.float().mean()
        out = tracer.step(front, b['block'], b['origin'], fibers)
        loss, st = step_losses(out, front, b['gt'], b['valid'], cfg)
        confs.append(torch.sigmoid(out.conf_logit.detach()))
        (loss/steps).backward()
        loss = loss.detach()
        total = total+loss/steps
        for k, v in st.items():
            stats[k] = stats.get(k, 0.)+v/steps
        front = tracer.advance(front, out)
        pos, far = perturb(front.pos, b['gt'], b['valid'], cfg, snap, generator)
        tracer.move(front, pos)
        resets += far.float().mean()/steps
    stats['reset'] = resets
    stats['jumped'] = torch.as_tensor(jumped)
    return total, stats, front, confs


def _pinned(batch: dict) -> dict:
    """The batch with its tensors pinned (packed blocks: their data; shapes stay on the host as they are)."""
    pin = lambda x: x.pin_memory() if isinstance(x, torch.Tensor) else x
    block = lambda p: dict(p, data=pin(p['data']))
    return dict({k: pin(v) for k, v in batch.items() if k != 'block'}, block=block(batch['block']))


def device_batches(batches, device, pinned_ahead: int = 2, timings: dict | None = None, pad_blocks: bool = True):
    """``evaluation.rollouts.to_device`` of each batch, the next batch's copy (and its block's conversion to float) already
    running on a side stream while the current one is in use, so the host-to-device copy overlaps compute.

    A background thread pins the batches (at most ``pinned_ahead`` waiting). Load the batches unpinned: a
    DataLoader's own pin thread pins every prefetched batch (prefetch_factor * workers of them, gigabytes each at
    level 0) and spends its time allocating page-locked memory instead of reusing it. ``timings`` (a dict) collects
    the background thread's seconds waiting for the loader (``loader``) and pinning (``pin``)."""
    if device.type != 'cuda':
        yield from (E.to_device(b, device, pad_blocks) for b in batches)
        return
    side, main = torch.cuda.Stream(device), torch.cuda.current_stream(device)
    ready_batches = queue.Queue(pinned_ahead)

    def produce():
        clock = time.perf_counter
        try:
            batches_iter, t0 = iter(batches), clock()
            for b in batches_iter:
                t1 = clock()
                pinned = _pinned(b)
                if timings is not None:
                    timings['loader'] = timings.get('loader', 0.)+t1-t0
                    timings['pin'] = timings.get('pin', 0.)+clock()-t1
                ready_batches.put(pinned)
                t0 = clock()
            ready_batches.put(None)
        except BaseException as e:  # hand loader errors to the consumer
            ready_batches.put(e)

    threading.Thread(target=produce, daemon=True).start()

    def load():
        b = ready_batches.get()
        if isinstance(b, BaseException):
            raise b
        if b is None:
            return None
        with torch.cuda.stream(side):
            return E.to_device(b, device, pad_blocks), side.record_event()

    pending = load()
    while pending is not None:
        b, ready = pending
        pending = load()
        main.wait_event(ready)
        blocks = b['block'] if isinstance(b['block'], list) else [b['block']]
        seeds = [b[k] for k in ('seed', 'seed_mask')+FIBER_KEYS if k in b]
        for x in [b['gt'], b['valid'], b['origin']]+blocks+seeds:
            x.record_stream(main)  # used on the main stream: not reusable before that work is done
        yield b


def fiber_shares(fraction: float, runs: dict, patches: dict) -> dict:
    """``data.fiber_fraction_by_runs``: per batch kind, ``fraction`` scaled by the kind's usable fiber runs per usable
    patch over the highest such density of any kind, so each fiber run is drawn no more often, relative to the
    patches, than at the front size with the most runs."""
    density = {k: runs[k]/max(1, patches[k]) for k in runs}
    top = max(density.values(), default=0.)
    return {k: fraction*rho/top if top else 0. for k, rho in density.items()}


def front_variants(train_cfg, index, pad: float) -> dict:
    """``data.front_points_range``: batch kind -> (training config, its episode sampler, batch size, share of
    batches). Kinds are the point counts N (an N-point front), plus ``'follow'`` with ``data.follow_fraction``: a
    3-point front following a fiber (``data.fiber_follow``, model mode 1). The batch holds about
    ``train.points_per_batch`` points; the block bound ``data.roi`` grows with the front so long fronts still fit."""
    d, tc = train_cfg.data, train_cfg.train
    per_batch = tc.points_per_batch or tc.batch*d.front_points
    counts = range(int(d.front_points_range[0]), int(d.front_points_range[1])+1)
    kinds = [(n, n, False) for n in counts]+([('follow', 3, True)] if d.follow_fraction else [])
    out = {}
    for kind, n, follow in kinds:
        cfg = copy.deepcopy(train_cfg)
        cfg.data.front_points, cfg.data.fiber_follow = n, follow
        if follow:
            cfg.data.fiber_fraction = 0.
        unit = 64 << d.volume_level
        cfg.data.roi = max(d.roi, -(-int((n-1)*d.point_spacing+2*pad+2*d.gt_margin)//unit)*unit)
        sampler = EpisodeSampler(cfg.data, index, 'train', d.steps, cfg.data.roi, pad)
        share = d.follow_fraction if follow else (1-d.follow_fraction)/len(counts)
        out[kind] = (cfg, sampler, max(1, round(per_batch/n)), share)
    if d.fiber_fraction_by_runs:  # the samplers read their (shared) data config when drawing
        sheet = [k for k in out if k != 'follow']
        shares = fiber_shares(d.fiber_fraction, {k: len(out[k][1].fiber_runs) for k in sheet},
                              {k: len(out[k][1].index) for k in sheet})
        for k, f in shares.items():
            out[k][0].data.fiber_fraction = f
    for kind, (cfg, sampler, size, share) in out.items():
        print(f'{kind}: {cfg.data.front_points} front points, batch {size}, {share:.0%} of batches, {len(sampler.index)} '
              f'usable patches, {len(sampler.fiber_runs)} fiber runs ({cfg.data.fiber_fraction:.1%} of episodes), '
              f'roi {cfg.data.roi}', flush=True)
    return out


def optimize_step(tracer: Tracer, opt, ema, b: dict, cfg, spec, generator, front=None, parallel=None):
    """One training iteration on batch ``b``: rollout, backward, gradient clipping, optimizer and EMA update.
    Returns ``train_rollout``'s loss, stats (plus the gradient norm ``grad``), final front and confidences.
    With ``parallel`` (shared/distributed.DataParallel) the gradients of every process's batch are averaged (each
    batch weighted equally) before clipping, so all processes apply the same update."""
    model, tc = tracer.model, cfg.train
    opt.zero_grad(set_to_none=True)
    loss, stats, front, confs = train_rollout(tracer, b, cfg, spec, generator, front)
    if parallel is not None:
        _, finite, _ = parallel.average_gradients(model, 1, math.isfinite(float(loss)))
        if not finite:
            raise FloatingPointError('Nonfinite training loss on a data-parallel process')
    stats['grad'] = torch.tensor(clip_training_gradients(model, tc.grad_clip)['grad_norm'])
    opt.step()
    if ema:
        ema.update(model)
    return loss, stats, front, confs


class EMA:
    """Exponential moving average of a model's weights (``train.ema``)."""

    def __init__(self, model, decay: float, updates: int = 0):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, updates

    def update(self, model) -> None:
        """One update of the average (``shared.training.update_ema``: the decay ramps up over the first updates)."""
        self.updates += 1
        update_ema(self.model, model, self.updates, self.decay)


def save(path, model, opt, cfg, it, best, minutes=0., ema: EMA | None = None):
    extra = dict(ema=ema.model.state_dict(), ema_updates=ema.updates) if ema else {}
    atomic_torch_save(dict(model=model.state_dict(), opt=opt.state_dict(), config=cfg.to_dict(), iteration=it,
                           best=best, minutes=minutes, **extra), path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', help='JSON config (path, or a name under configs/)')
    ap.add_argument('--name')
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.FIELD=VALUE')
    ap.add_argument('--resume', help='checkpoint to resume (uses its config; --set still applies)')
    ap.add_argument('--init', help='start from these model weights (architecture must match the config); fresh '
                                   'optimizer, schedule and iteration count, unlike --resume')
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # validation lines reach a redirected log as they happen

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        document = checkpoint['config']
        for text in args.set:
            section, name, value = C.parse_override(text)
            document[section][name] = value
        cfg = C.from_dict(document)
    else:
        overrides = list(args.set)+([f'train.name={args.name}'] if args.name else [])
        cfg = C.load(args.config, overrides)
    tc, d = cfg.train, cfg.data
    parallel = from_environment(tc.gpus)  # torchrun processes when train.gpus > 1 (shared/distributed.py)
    main_rank = parallel is None or parallel.is_main
    rank_seed = parallel.seed_offset() if parallel is not None else 0  # each process draws its own data
    run = prepare_run_dir(cfg.run_dir.parent, cfg.run_dir.name, resume=checkpoint is not None) if main_rank else cfg.run_dir
    if parallel is not None:
        parallel.barrier()
    if main_rank:
        (run/'config.json').write_text(json.dumps(cfg.to_dict(), indent=2))
    raise_open_file_limit()
    # Loader workers are forkserver children, not ours: exit through Python on SIGTERM so the DataLoader
    # shuts them down instead of leaving them running.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    torch.manual_seed(tc.seed)
    if tc.compile_geometry:
        from vesuvius.neural_tracing.tracers.sheets.geometry import use_compiled_projection
        use_compiled_projection()
    device = torch.device(parallel.device if parallel is not None else tc.device)

    for turn in range(parallel.world_size if parallel is not None else 1):  # one process at a time builds the cache
        if parallel is None or parallel.rank == turn:
            index = build_index(d.patches_dir, C.OUTPUT/'cache')
        if parallel is not None:
            parallel.barrier()
    pad = episode_pad(cfg.model, tc)
    train_cfg = cfg  # training episodes start history_rows further back (memory warm-up); validation's do not
    if tc.history_rows:
        train_cfg = copy.deepcopy(cfg)
        train_cfg.data.history += tc.history_rows
    train_sampler = EpisodeSampler(train_cfg.data, index, 'train', d.steps, d.roi, pad)
    val_sampler = EpisodeSampler(d, index, 'val', d.steps, d.roi, pad)
    train_spec, spec = train_sampler.spec, val_sampler.spec
    print(f'{len(train_sampler.index)} usable patches; GT grid {train_spec.Gu}x{train_spec.Gv}; '
          f'blocks within {d.roi} vx at level {d.volume_level}', flush=True)
    val_set = None  # only rank 0 validates
    if main_rank:
        t0 = time.time()
        val_set = fixed_episodes(val_sampler, tc.val_episodes, tc.seed)
        print(f'{len(val_set)} validation episodes in {time.time()-t0:.1f} s', flush=True)
    jobs = None
    if tc.live_chains:  # the workers serve live-chain continuations as whole batches
        jobs = multiprocessing.Queue(64)
        jobs.cancel_join_thread()  # never block exit on unread jobs
    variants = front_variants(train_cfg, index, pad) if d.front_points_range else None
    seed = tc.seed+(checkpoint['iteration'] if checkpoint else 0)+rank_seed
    if variants:
        stream = MixedFrontStream({k: v[1] for k, v in variants.items()}, {k: v[2] for k, v in variants.items()}, seed,
                                  {k: v[3] for k, v in variants.items()})
    else:
        stream = EpisodeStream(train_sampler, seed, batch=tc.batch if jobs else 0, jobs=jobs)
    batched = jobs is not None or variants is not None  # the stream yields whole batches
    loader = torch.utils.data.DataLoader(
        stream, batch_size=None if batched else tc.batch, num_workers=tc.workers,
        collate_fn=keep_batch if batched else collate,
        pin_memory=device.type == 'cuda' and jobs is not None,  # otherwise device_batches pins
        persistent_workers=tc.workers > 0, prefetch_factor=tc.prefetch if tc.workers else None)

    if checkpoint and legacy_state(checkpoint['model']):
        raise ValueError(f'{args.resume} predates the shared transformer layer (models/blocks.py), so its optimizer '
                         'state does not match the model: start a new run from its weights with --init')
    model = build_model(cfg).to(device)
    scales = [p for n, p in model.named_parameters() if n.endswith('.logit_scale')]
    groups = [dict(params=[p for n, p in model.named_parameters() if not n.endswith('.logit_scale')])]
    if scales:  # attention logit scales (model.qk_logit_scale): no decay, a faster LR, as in the fiber tracer
        groups.append(dict(params=scales, weight_decay=0., lr_scale=LOGIT_SCALE_LR_SCALE))
    opt = torch.optim.AdamW(groups, lr=tc.lr, weight_decay=tc.weight_decay)
    start, best = 0, -1.
    if args.init and not checkpoint:
        saved = torch.load(args.init, map_location='cpu', weights_only=False)
        init_model_state(model, saved.get('ema') or saved['model'])  # the averaged weights, when the run had them
        print(f"initialised weights from {args.init}{' (EMA)' if saved.get('ema') else ''}", flush=True)
    if checkpoint:
        load_model_state(model, checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        start, best = checkpoint['iteration'], checkpoint['best']
    print(f'model parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f} M', flush=True)
    ema = None
    if tc.ema:
        ema = EMA(model, tc.ema)
        if checkpoint and checkpoint.get('ema'):
            load_model_state(ema.model, checkpoint['ema'])
            ema.updates = checkpoint.get('ema_updates', 0)
    if parallel is not None:  # identical starting weights on every process, then each draws its own randomness
        for net in (model, ema.model) if ema else (model,):
            parallel.broadcast_module(net)
        torch.manual_seed(tc.seed+(checkpoint['iteration'] if checkpoint else 0)+rank_seed)
    if tc.compile_model:
        for net in (model, ema.model) if ema else (model,):
            compile_model(net)
    tracer = Tracer(model, cfg, device)
    tracers = {n: Tracer(model, v[0], device) for n, v in variants.items()} if variants else {}
    eval_tracer = Tracer(ema.model, cfg, device) if ema else tracer  # validation uses the averaged weights
    generator = torch.Generator(device=device)
    generator.manual_seed(tc.seed+start+rank_seed)
    chains = LiveChains(train_cfg, train_spec, tracer, jobs, generator) if jobs else None

    log_path = run/'log.jsonl'
    log_rows = [json.loads(line) for line in log_path.read_text().splitlines()] if checkpoint and log_path.exists() else []
    if not checkpoint:
        log_path.write_text('')

    def write(row):
        if not main_rank:
            return
        # Reopen per row: tools that swap files under output/ (git stash/restore, sync) must not swallow the log.
        log_rows.append(row)
        with log_path.open('a') as fh:
            fh.write(json.dumps(row)+'\n')

    def validate(it):
        viz = bool(tc.viz_every and it % tc.viz_every == 0)
        per_step, records = E.rollout_metrics(eval_tracer, val_set, cfg, spec, d.steps, device, tc.batch,
                                              records=4 if viz else 0,
                                              surfaces=run/f'val_{it:06d}'/'tifxyz' if viz else None,
                                              surface_count=8)
        summary = E.summarize(per_step)
        write(dict(kind='val', iteration=it, **summary))
        if records:
            out = run/f'val_{it:06d}'
            out.mkdir(exist_ok=True)
            for i, record in enumerate(records):
                episode_figure(record, cfg, out/f'episode_{i:02d}.png')
            metric_figure({'val': per_step}, out/'metrics.png')
            training_figure(log_rows, run/'training.png')
        return summary

    started, data_time, it = time.time(), 0., start
    resumed_minutes = float(checkpoint.get('minutes', 0.)) if checkpoint else 0.
    # Every process follows rank 0's clock (the learning rate and the time budget must agree across processes).
    elapsed_minutes = lambda: resumed_minutes+(time.time()-started)/60
    minutes = elapsed_minutes if parallel is None else lambda: parallel.broadcast_value(elapsed_minutes())
    world = parallel.world_size if parallel is not None else 1
    window = {}
    bar = tqdm(range(start, tc.iterations), initial=start, total=tc.iterations, dynamic_ncols=True,
               disable=not main_rank)
    # Live chains decide what the loader serves from finished rollouts, so their batches are not fetched ahead.
    batches = ((E.to_device(b, device) for b in loader) if chains else
               device_batches(loader, device, pad_blocks=not d.unpadded_blocks))
    for it in bar:
        t_data = time.time()
        while True:  # with live chains, skip continuations whose chains were dropped
            b = next(batches)
            b, chain = chains.begin(b, it) if chains else (b, None)
            if chains is None or chain is not None:
                break
        data_time += time.time()-t_data
        now = minutes()
        for group in opt.param_groups:
            group['lr'] = learning_rate(it, tc, now)*group.get('lr_scale', 1.)
        if variants:  # this batch's kind: front size, or fiber following
            kind = b['front_points']
            vcfg = variants[kind][0]
            loss, stats, front, confs = optimize_step(tracers[kind], opt, ema, b, vcfg, variants[kind][1].spec,
                                                      generator, parallel=parallel)
            stats['front_points'] = torch.tensor(float(vcfg.data.front_points))
            stats['follow'] = torch.tensor(float(vcfg.data.fiber_follow))
        else:
            loss, stats, front, confs = optimize_step(tracer, opt, ema, b, train_cfg, train_spec, generator,
                                                      chain['front'] if chain else None, parallel=parallel)
        if chains:
            chains.finish(b, chain, front, confs, it)
            stats.update(chains.stats)
        for k, v in stats.items():
            window.setdefault(k, []).append(v)
        done = it+1
        if done % tc.log_every == 0 and main_rank:  # rank 0's batches stand for the iteration's
            row = {k: float(torch.stack([torch.as_tensor(x) for x in v]).float().mean()) for k, v in window.items()}
            elapsed = time.time()-started
            row.update(kind='train', iteration=done, lr=learning_rate(it, tc, elapsed_minutes()),
                       minutes=elapsed_minutes(), episodes_per_s=(done-start)*tc.batch*world/elapsed, data_fraction=data_time/elapsed)
            write(row)
            window = {}
            bar.set_postfix_str(f"loss {row['loss']:.3f} dist {row['dist']:.2f} same {row['same_sheet']:.2f} "
                                f"conf {row['confident']:.2f} prec {row['conf_precision']:.2f} "
                                f"caught {row['off_caught']:.2f} "
                                f"{row['episodes_per_s']:.0f} ep/s data {row['data_fraction']:.0%}")
        if main_rank and (done % tc.val_every == 0 or done == tc.iterations):
            summary = validate(done)
            tqdm.write(f"[{done}] val score {summary['score_mean']:.3f} coverage {summary['coverage_mean']:.3f} "
                       f"kept_off {summary['kept_off_mean']:.3f} caught {summary['caught_mean']:.3f} "
                       f"same_sheet {summary['same_sheet_mean']:.3f} "
                       f"precision {summary['precision_mean']:.3f} rejoined {summary['rejoined_mean']:.3f} "
                       f"dist p90 {summary['dist_p90_mean']:.2f}")
            if summary['score_mean'] > best:
                best = summary['score_mean']
                save(run/'best.pt', model, opt, cfg, done, best, elapsed_minutes(), ema)
        if main_rank and done % tc.ckpt_every == 0:
            save(run/'last.pt', model, opt, cfg, done, best, elapsed_minutes(), ema)
        if main_rank and tc.keep_every and done % tc.keep_every == 0:
            save(run/f'ckpt_{done:06d}.pt', model, opt, cfg, done, best, elapsed_minutes(), ema)
        if tc.max_minutes and minutes() > tc.max_minutes:  # the same decision on every process (rank 0's clock)
            if main_rank:
                tqdm.write(f'time budget reached at iteration {done}')
            if main_rank and done % tc.val_every:
                summary = validate(done)
                if summary['score_mean'] > best:
                    best = summary['score_mean']
                    save(run/'best.pt', model, opt, cfg, done, best, elapsed_minutes(), ema)
            break
    del batches, loader  # stop the loader workers before evaluating
    if parallel is not None:
        parallel.close()
    if not main_rank:
        return
    save(run/'last.pt', model, opt, cfg, it+1, best, elapsed_minutes(), ema)
    training_figure(log_rows, run/'training.png')
    if cfg.eval.after_training:
        final_eval(cfg, run, device)


def final_eval(cfg, run, device):
    """Short free rollouts of the best model on held-out episodes."""
    e = cfg.eval
    t0 = time.time()
    episodes = E.eval_episodes(cfg, e.steps, e.roi, e.episodes)
    best_model, _, _ = load_checkpoint(run/'best.pt', device)
    report = E.run(Tracer(best_model, cfg, device), cfg, run/'eval', episodes, e.steps, e.roi, records=4)
    s = report['summary']
    print(f"eval: coverage {s['coverage_mean']:.3f} (last {s['coverage_last']:.3f}) same_sheet "
          f"{s['same_sheet_mean']:.3f} precision {s['precision_mean']:.3f} caught {s['caught_mean']:.3f} "
          f"rejoined {s['rejoined_mean']:.3f}", flush=True)
    print(f'final eval: {e.episodes} episodes x {e.steps} steps in {time.time()-t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
