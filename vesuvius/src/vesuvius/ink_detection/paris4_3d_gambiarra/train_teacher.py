"""Train the DINO point teacher of ink_paris4_3d_4.8um_gambiarra, one stage per call.

    P=vesuvius.ink_detection.paris4_3d_gambiarra
    python -m $P.train_teacher --stage linear20k   --ssl-checkpoint SSL.pt --out runs/linear20k
    python -m $P.train_teacher --stage scribble5k  --ssl-checkpoint SSL.pt --out runs/scribble5k  --init runs/linear20k/step-020000.pth
    python -m $P.train_teacher --stage scribble10k --ssl-checkpoint SSL.pt --out runs/scribble10k --init runs/scribble5k/step-005000.pth
    python -m $P.train_teacher --stage all_aug     --ssl-checkpoint SSL.pt --out runs/all_aug     --init runs/scribble10k/step-010000.pth

Add --resume <out>/step-XXXXXX.pth to continue an interrupted stage (same CT, labels and SSL checkpoint; checked).
Resuming the stopping checkpoint runs no update; at the stage endpoint it repairs COMPLETE.json. Every stage takes 16 views
per update (8 anchors x 2 views), AdamW (weight decay .01), gradient-norm clip 1 and the plan streams of sampling.py.

  linear20k    all scribble samples; backbone + linear head; rotated, photometric views from the 320^3 context. lr: head .01,
               backbone 3e-6, 50-update warmup, cosine to .1x over a 40,000-update horizon, stopped at 20,000.
  scribble5k   all scribble samples; adds the voxel residual and the weak tube sites (BCE weight 1). lr: head + residual 3e-4,
               backbone 1e-6 (frozen for 250 updates, then a 50-update ramp), cosine to .1x at 5,000.
  scribble10k  continues scribble5k (weights, AdamW moments, plan RNG): updates 5,001-10,000, cosine from
               head 3e-5 / backbone 1e-7 to .1x.
  all_aug      all 9,627 labels; restores scribble10k, then fresh anchor decks; views rotated and zoomed (rotation_aug).
               lr: 50-update re-warmup to head 3e-4 / backbone 1e-6, cosine to .1x at 5,000.

--init also accepts the original run checkpoints (their 'delta' / 'trainable' names are converted).
"""
import argparse
import copy
import math
import time

import numpy as np

from vesuvius.ink_detection.paris4_3d_gambiarra import runtime as rt  # sets CUBLAS_WORKSPACE_CONFIG before torch initializes CUDA
import torch

from vesuvius.ink_detection.paris4_3d_gambiarra import rotation_aug, sampling, views
from vesuvius.ink_detection.paris4_3d_gambiarra.ct import ChunkStore, PARIS4_LEVEL0
from vesuvius.ink_detection.paris4_3d_gambiarra.data import ClickData
from vesuvius.ink_detection.paris4_3d_gambiarra.labels import resolve_scribbles
from vesuvius.ink_detection.paris4_3d_gambiarra.losses import balanced_point_bce, weak_bce
from vesuvius.ink_detection.paris4_3d_gambiarra.teacher import Teacher, convert_original, load_backbone, SSL_CHECKPOINT_SHA256

SEED = 27
PLAN_SEED = 2027  # historical plan seed
STAGES = dict(
    linear20k=dict(labels='all', residual=False, checkpoint='half', start=0, end=20000, horizon=40000,
                   saves=(2000, 5000, 7500, 10000, 15000, 20000), init=None),
    scribble5k=dict(labels='all', residual=True, checkpoint='all', start=0, end=5000, horizon=5000,
                    saves=(1000, 2500, 5000), init='linear20k'),
    scribble10k=dict(labels='all', residual=True, checkpoint='all', start=5000, end=10000, horizon=10000,
                     saves=(6000, 7500, 10000), init='scribble5k'),
    all_aug=dict(labels='all', residual=True, checkpoint='all', start=0, end=5000, horizon=5000,
                 saves=(1000, 2500, 5000), init='scribble10k'),
)


def cosine(step, start, end, warmup):
    """Linear warmup over `warmup` updates after `start`, then cosine from 1 to .1 at `end`."""
    if warmup and step - start <= warmup:
        return (step - start) / warmup
    return .1 + .9 * .5 * (1 + math.cos(math.pi * (step - start - warmup) / (end - start - warmup)))


def schedule(stage, opt, step):
    if stage == 'linear20k':
        for group in opt.param_groups:
            group['lr'] = group['base_lr'] * cosine(step, 0, 40000, 50)
    elif stage == 'scribble5k':
        fac = cosine(step, 0, 5000, 50)
        for group in opt.param_groups:
            scale = fac
            if group['role'] == 'backbone':
                scale = 0. if step <= 250 else min(1., (step - 250) / 50) * fac
            group['lr'] = group['base_lr'] * scale
    elif stage == 'scribble10k':
        fac = cosine(step, 5000, 10000, 0)
        for group in opt.param_groups:
            group['lr'] = dict(head=3e-5, backbone=1e-7)[group['role']] * fac
    else:
        fac = cosine(step, 0, 5000, 50)
        for group in opt.param_groups:
            group['lr'] = dict(head=3e-4, backbone=1e-6)[group['role']] * fac


def optimizer(stage, teacher):
    backbone = [p for p in teacher.backbone.parameters() if p.requires_grad]
    if stage == 'linear20k':
        groups = [dict(params=list(teacher.head.parameters()), lr=.01, base_lr=.01),
                  dict(params=backbone, lr=3e-6, base_lr=3e-6)]
    else:
        groups = [dict(params=list(teacher.head.parameters()) + list(teacher.residual.parameters()), lr=3e-4, base_lr=3e-4, role='head'),
                  dict(params=backbone, lr=0., base_lr=1e-6, role='backbone')]
    return torch.optim.AdamW(groups, weight_decay=.01)


def trainable_of(state):
    """Trainable tensors of a port or original checkpoint, in this package's names."""
    return convert_original(state['trainable'] if 'trainable' in state else state['delta'])


class Loader:
    """CPU part of a plan's packet (crops, labels, normalization limits); run in the lookahead thread."""
    def __init__(self, stage, data):
        self.stage, self.data = stage, data

    def __call__(self, plan, step):
        d = self.data
        if self.stage == 'all_aug':
            geo = rotation_aug.geometry(d, plan, step)
            parents = np.stack([d.ct.parent(item['anchor']) for item in plan['plans']])
            norm = [list(d.limits(item['anchor'])) for item in plan['plans']]
            return dict(plan=plan, step=step, parents=parents, norm=norm, geo=geo,
                        ids=[q['ids'] for q in geo], coordinates=[q['coordinates'] for q in geo],
                        weak_ids=[q['weak_ids'] for q in geo], weak_coordinates=[q['weak_coordinates'] for q in geo],
                        weak_labels=[q['weak_labels'] for q in geo])
        packet = dict(plan=plan, step=step, ids=[], coordinates=[], norm=[], transforms=[], images=[],
                      weak_ids=[], weak_coordinates=[], weak_labels=[])
        for item in plan['plans']:
            i = item['anchor']
            t = d.transform(item)
            ids, coords = d.clicks(item, t)
            if self.stage == 'linear20k':
                context = d.ct.context(i)
                packet['images'].append(context)
                packet['norm'].append(list(d.ct.limits(i, raw=context[32:288, 32:288, 32:288])))
            else:
                packet['images'].append(views.cube_crop(d.ct.parent(i), item))
                packet['norm'].append(list(d.limits(i)))
                wi, wq, wy = d.weak_sites(item, t)
                packet['weak_ids'].append(wi)
                packet['weak_coordinates'].append(wq)
                packet['weak_labels'].append(wy)
            packet['ids'].append(ids)
            packet['coordinates'].append(coords)
            packet['transforms'].append(t)
        packet['images'] = np.stack(packet['images'])
        return packet


def make_views(stage, packet):
    photos = [item['photo'] for item in packet['plan']['plans']]
    if stage == 'linear20k':
        return views.rotated_views(packet['images'], packet['norm'], packet['transforms'], photos), None
    if stage == 'all_aug':
        return rotation_aug.views(packet['parents'], packet['norm'], packet['geo'])
    return views.cube_views(packet['images'], packet['norm'], photos), None


def linear_loss(teacher, x, packet, labels, micro=8):
    """Stage 1: the linear readout at every visible training point; balanced BCE over all 16 views."""
    logits, ids = [], []
    for start in range(0, len(x), micro):
        tokens = teacher.tokens(x[start:start + micro])
        for j in range(len(tokens)):
            q = torch.as_tensor(packet['coordinates'][start + j], device=x.device)
            logits.append(teacher.readout(tokens, q, index=j))
            ids.extend(packet['ids'][start + j].tolist())
    ids = torch.tensor(ids, device=x.device)
    bce, unique = balanced_point_bce(torch.cat(logits), ids, torch.as_tensor(labels, device=x.device)[ids])
    return bce, dict(bce=bce.detach(), weak_bce=bce.detach() * 0, weak_count=0, unique=unique)


def decoder_loss(teacher, x, packet, labels):
    """Stages 2-4: readout + residual at the training points (balanced BCE) plus 1.0 x the weak-site BCE."""
    scores, ids, weak_scores, weak_ids, weak_labels = [], [], [], [], []
    for start in range(0, len(x), 2):
        tokens = teacher.tokens(x[start:start + 2])
        for k in range(len(tokens)):
            j = start + k
            tok = tokens[k:k + 1]
            field = teacher.field(tok, x[j:j + 1])
            q = torch.as_tensor(packet['coordinates'][j], device=x.device, dtype=torch.float32)
            scores.append(teacher.query(tok, field, q))
            ids.extend(packet['ids'][j].tolist())
            if len(packet['weak_ids'][j]):
                qq = torch.as_tensor(packet['weak_coordinates'][j], device=x.device, dtype=torch.float32)
                weak_scores.append(teacher.query(tok, field, qq))
                weak_ids.extend(packet['weak_ids'][j].tolist())
                weak_labels.extend(packet['weak_labels'][j].tolist())
    ids = torch.tensor(ids, device=x.device)
    bce, unique = balanced_point_bce(torch.cat(scores), ids, torch.as_tensor(labels, device=x.device)[ids])
    weak, count = bce * 0, 0
    if weak_scores:
        weak, count = weak_bce(torch.cat(weak_scores), torch.tensor(weak_ids, device=x.device),
                               torch.tensor(weak_labels, device=x.device))
    return bce + 1.0 * weak, dict(bce=bce.detach(), weak_bce=weak.detach(), weak_count=count, unique=unique)


def aug_row(packet, photo_attempts):
    geo = packet['geo']
    return dict(labels=int(sum(q['labels'] for q in geo)), weak_labels=int(sum(q['weak'] for q in geo)),
                plan_labels=int(sum(q['plan_labels'] for q in geo)), tube_plan=int(sum(q['tube_plan'] for q in geo)),
                tube_rotated=int(sum(q['tube_rotated'] for q in geo)),
                attempts_mean=float(np.mean([q['attempts'] for q in geo])), zoom_on=int(sum(q['zoom'] != 1. for q in geo)),
                angle_mean=float(np.mean([q['angle'] for q in geo])),
                photo_attempts_mean=float(np.mean(photo_attempts)), photo_attempts_max=int(max(photo_attempts)))


def input_identity(store, scribbles):
    """The inputs a resumed run must share with its checkpoint: CT array, VC3D scribbles, SSL backbone
    (load_backbone accepts only this sha256)."""
    return dict(ct=store.identity(), labels='all', points_sha256=rt.sha256_file(scribbles),
                ssl_checkpoint_sha256=SSL_CHECKPOINT_SHA256)


def run(args):
    assert not (args.stage == 'linear20k' and args.init), 'linear20k starts from --ssl-checkpoint, not --init'
    out = rt.output_directory(args.out, resume=bool(args.resume))
    rt.configure_torch(SEED)
    args.scribbles = resolve_scribbles(args.scribbles)
    store = ChunkStore(args.ct, cache_gib=args.chunk_cache_gib)
    try:
        train(args, out, store)
    finally:
        store.close()


def train(args, out, store):
    spec = STAGES[args.stage]
    stage = args.stage
    inputs = input_identity(store, args.scribbles)
    stop = min(spec['end'], args.stop_at or spec['end'])
    resumed = None
    if args.resume:
        resumed = rt.load(args.resume)
        if resumed.get('stage') != stage:
            raise ValueError(f'{args.resume} is not a {stage} checkpoint')
        rt.check_inputs(resumed, inputs, args.resume)
    step = spec['start'] if resumed is None else resumed['step']
    if step > stop or (step == stop and resumed is None):
        raise ValueError(f'nothing to do: checkpoint step {step}, stop {stop}')
    if step == stop:  # the checkpoint holds the stopping update: no update, only the completion marker
        rt.write_json(out / 'RESUMED.json', dict(stage=stage, start_step=step, stop_step=stop, updates=0,
                                                 resume=args.resume, inputs=inputs, init=resumed.get('init'), time=time.time()))
        if stop == spec['end']:
            rt.write_json(out / 'COMPLETE.json', dict(stage=stage, step=stop, checkpoint=str(args.resume), repaired=True))
        return
    data = ClickData(store, weak=spec['residual'], cache_gib=args.cache_gib, points_json=args.scribbles)
    teacher = Teacher(load_backbone(args.ssl_checkpoint), spec['residual'], spec['checkpoint'], seed=SEED)
    opt = optimizer(stage, teacher)
    rng = np.random.default_rng(PLAN_SEED)
    deck = sampling.ClassDeck(data.train, data.y, rng)
    draw = sampling.draw_rotated if stage == 'linear20k' else sampling.draw_cube

    def restore(state, full=True):
        teacher.load_trainable(trainable_of(state))
        if full:
            opt.load_state_dict(state['optimizer'])
            rt.set_rng_state(rng, state['numpy_rng'])
            deck.left = copy.deepcopy(state['decks'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state(state['cuda_rng'])

    init = None
    if resumed is not None:
        restore(resumed)
        pending, init = resumed['pending_plan'], resumed.get('init')
        del resumed
    else:
        if spec['init'] is not None:
            assert args.init, f'--init (a {spec["init"]} checkpoint) is required for {stage}'
            state = rt.load(args.init)
            assert state['step'] == dict(scribble5k=20000, scribble10k=5000, all_aug=10000)[stage], state['step']
            if 'stage' in state:
                assert state['stage'] == spec['init'], state['stage']
            else:  # original run checkpoints
                assert state['fold'] == 2
                assert state['head_type'] == 'linear' if stage == 'scribble5k' else state['arm'] == 'scribble'
            if stage == 'scribble5k':
                # The linear20k weights; the residual starts from its seed-27 initialization, the optimizer fresh.
                trainable = trainable_of(state)
                current = teacher.trainable_state()
                assert set(trainable) == {n for n in current if not n.startswith('residual.')}
                teacher.load_trainable({**current, **trainable})
            else:
                restore(state)
                assert state['pending_plan'] is None
            del state
            init = dict(path=str(args.init), sha256=rt.sha256_file(args.init))
        if stage == 'all_aug':
            deck.refresh()  # fresh decks over all 9,627 points from the restored plan RNG
        pending = draw(rng, deck, data.q, data.parents, data.eval_parents)
    if args.resume:
        rt.truncate_jsonl(out / 'curve.jsonl', step)
        (out / 'COMPLETE.json').unlink(missing_ok=True)
    rt.write_json(out / ('RESUMED.json' if args.resume else 'STARTED.json'), dict(
        stage=stage, start_step=step, stop_step=stop, end_step=spec['end'], labels=spec['labels'],
        train_points=len(data.train), heldout_points=len(data.heldout), init=init, resume=args.resume, ct=str(args.ct),
        ssl_checkpoint=str(args.ssl_checkpoint), ssl_checkpoint_sha256=SSL_CHECKPOINT_SHA256, inputs=inputs,
        weak_audit=getattr(data, 'weak_audit', None), time=time.time()))
    loss_fn = linear_loss if stage == 'linear20k' else decoder_loss
    clip = list(teacher.parameters()) if stage == 'linear20k' else [p for p in teacher.parameters() if p.requires_grad]
    look = rt.Lookahead(Loader(stage, data))
    look.submit(pending, step + 1)
    started = time.time()
    try:
        for number in range(step + 1, stop + 1):
            begin = time.time()
            packet, wait = look.take()
            if number <= spec['start'] + 8:
                rt.write_json(out / f'plans/PLAN-{number:06d}.json', packet['plan'])
            # The next plan is drawn before this update (the original order); stage 1 kept drawing up to its
            # 40,000-update horizon.
            pending = draw(rng, deck, data.q, data.parents, data.eval_parents) if number < spec['horizon'] else None
            if pending is not None and number < stop:
                look.submit(pending, number + 1)
            x, photo_attempts = make_views(stage, packet)
            teacher.backbone_active = stage != 'scribble5k' or number > 250
            schedule(stage, opt, number)
            opt.zero_grad(set_to_none=True)
            value, info = loss_fn(teacher, x, packet, data.y)
            assert torch.isfinite(value)
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(clip, 1, error_if_nonfinite=True)
            opt.step()
            row = dict(step=number, loss=float(value.detach()), bce=float(info['bce']), weak_bce=float(info['weak_bce']),
                       weak_count=info['weak_count'], unique_points=len(info['unique']), gradient_norm=float(norm),
                       backbone_active=teacher.backbone_active, lr=opt.param_groups[0]['lr'],
                       backbone_lr=opt.param_groups[1]['lr'], data_wait=wait, step_seconds=time.time() - begin,
                       seconds=time.time() - started)
            if stage == 'all_aug':
                row.update(aug_row(packet, photo_attempts))
            rt.append_jsonl(out / 'curve.jsonl', row)
            del value, info, x, packet
            if number in spec['saves'] or number == stop:
                rt.save(out / f'step-{number:06d}.pth', dict(
                    stage=stage, step=number, trainable=teacher.trainable_state(), optimizer=rt.cpu_copy(opt.state_dict()),
                    numpy_rng=rt.rng_state(rng), decks=copy.deepcopy(deck.left), pending_plan=copy.deepcopy(pending),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), labels=spec['labels'],
                    inputs=inputs, init=init))
        if stop == spec['end']:
            rt.write_json(out / 'COMPLETE.json', dict(stage=stage, step=stop, checkpoint=str(out / f'step-{stop:06d}.pth'),
                                                      seconds=time.time() - started))
    finally:
        look.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=tuple(STAGES), required=True)
    p.add_argument('--ssl-checkpoint', required=True, help='dinovol_2 checkpoint (HF scrollprize/dinovol_v2_ps8_with_paris4_352500)')
    p.add_argument('--out', required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--init', help='checkpoint of the previous stage')
    source.add_argument('--resume', help='a checkpoint of this stage written by this script (same inputs)')
    p.add_argument('--ct', default=PARIS4_LEVEL0, help='PHercParis4 level-0 zarr array (URL or local directory)')
    p.add_argument('--scribbles', help='local VC3D ink/noink JSON (default: download released scribbles from Hugging Face)')
    p.add_argument('--cache-gib', type=float, default=16., help='decoded parent/context cache')
    p.add_argument('--chunk-cache-gib', type=float, default=4.)
    p.add_argument('--stop-at', type=int, help='save and stop after this update (smoke tests)')
    run(p.parse_args())


if __name__ == '__main__':
    main()
