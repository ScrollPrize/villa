"""Distil the released 3D U-Net student (ink_paris4_3d_4.8um_gambiarra) from the frozen DINO teacher.

    P=vesuvius.ink_detection.paris4_3d_gambiarra
    python -m $P.train_student --stage 1 --ssl-checkpoint SSL.pt --teacher runs/scribble10k/step-010000.pth --out runs/student1
    python -m $P.train_student --stage 2 --ssl-checkpoint SSL.pt --teacher runs/all_aug/step-005000.pth --out runs/student2 \
        --init runs/student1/train-state-020000.pth

Add --resume <out>/train-state-XXXXXX.pth to continue an interrupted stage (same CT, labels, pool, teacher and SSL
checkpoint; checked). A resume first rewrites the checkpoint's step-XXXXXX.safetensors export; resuming the stopping
checkpoint runs no update and, at the stage endpoint, repairs COMPLETE.json. One update ("lr2_kd4_u16"):

  16 labelled views (8 anchors x 2, sampling.draw_cube) and 16 unlabelled views (one per pool parent,
  assets/unlabelled_pool.csv 'train' rows). The student sees each normalized 128^3 view average-pooled to 64^3
  (4.8 um voxels); its 64^3 logits are resampled to the 128^3 native voxels.
  loss = sum over labelled views of
             sum_i w_i BCE(student(q_i), y_i)                 (each class weighs .5, each point equally within it)
           + 4 * mean_s BCE(student(s), teacher(s)) / 16      (KD on a 32^3 lattice minus sites within 1 voxel of a click)
       + sum over unlabelled views of 4 * mean_s BCE(student(s), teacher(s)) / 16   (all 32,768 lattice sites)
  teacher(s) = sigmoid of the frozen teacher's logit on the same view. AdamW (weight decay .01), clip 1, microbatch 8.

  stage 1  updates 1-20,000; all scribble samples; teacher scribble10k; lr 3e-4 with a 50-update
           warmup and cosine to .1x at 20,000.
  stage 2  updates 20,001-25,000; restores stage 1 at 20,000 (weights, AdamW, unlabelled stream); a new labelled
           stream over all 9,627 points (seed 2026092612); teacher all_aug; constant lr 3e-5. Released: 25,000.

Every stage uses all samples in the VC3D JSON. The historical run accumulated labels between stages.
"""
import argparse
import copy
import time
from collections import defaultdict

import numpy as np

from vesuvius.ink_detection.paris4_3d_gambiarra import runtime as rt  # sets CUBLAS_WORKSPACE_CONFIG before torch initializes CUDA
import torch
from torch.nn import functional as F

from vesuvius.ink_detection.paris4_3d_gambiarra import sampling, views
from vesuvius.ink_detection.paris4_3d_gambiarra.ct import ChunkStore, PARIS4_LEVEL0, Parents, sha256
from vesuvius.ink_detection.paris4_3d_gambiarra.data import ClickData
from vesuvius.ink_detection.paris4_3d_gambiarra.labels import resolve_scribbles, POOL_CSV, read_pool
from vesuvius.ink_detection.paris4_3d_gambiarra.losses import point_weights
from vesuvius.ink_detection.paris4_3d_gambiarra.student import Student, save_weights
from vesuvius.ink_detection.paris4_3d_gambiarra.teacher import Teacher, load_backbone, voxel_sample, SSL_CHECKPOINT_SHA256
from vesuvius.ink_detection.paris4_3d_gambiarra.train_teacher import cosine, input_identity, trainable_of

SEED = 27
UNLABELLED_SEED = 2026092302
MICROBATCH, KD, UKD = 8, 4., 4.
STAGES = {
    1: dict(labels='all', start=0, end=20000, click_seed=2027, saves=(500, 1000, 2000, 3500, 5000, 7500, 10000, 15000, 20000)),
    2: dict(labels='all', start=20000, end=25000, click_seed=2026092612, saves=(21000, 22500, 25000)),
}


def learning_rate(stage, step):
    return 3e-4 * cosine(step, 0, 20000, 50) if stage == 1 else 3e-5


class Loader:
    """CPU part of one update: labelled crops, labels and KD sites; unlabelled crops and sites."""
    def __init__(self, data, pool, pool_ct):
        self.data, self.pool, self.pool_ct, self.checked = data, pool, pool_ct, set()

    def pool_parent(self, i):
        raw = self.pool_ct.parent(i)
        if i not in self.checked:
            assert sha256(raw) == self.pool[i]['raw256_sha256'], f'pool parent {i}: CT bytes differ from the recorded sha256'
            self.checked.add(i)
        return raw

    def __call__(self, plan, uplan):
        d = self.data
        out = dict(plan=plan, uplan=uplan, crops=[], norm=[], ids=[], coordinates=[], sites=[], ucrops=[], unorm=[], usites=[])
        for item in plan['plans']:
            i = item['anchor']
            t = d.transform(item)
            ids, coords = d.clicks(item, t)
            q, keep = sampling.lattice(item, d.mask_clicks(item, t))
            out['crops'].append(views.cube_crop(d.ct.parent(i), item))
            out['norm'].append(list(d.limits(i)))
            out['ids'].append(ids)
            out['coordinates'].append(coords)
            out['sites'].append(q[keep])
        for item in uplan['plans']:
            i = item['unlabelled']
            q, keep = sampling.lattice(item, np.empty((0, 3), np.float32))
            out['ucrops'].append(views.cube_crop(self.pool_parent(i), item))
            out['unorm'].append([self.pool[i]['low'], self.pool[i]['high']])
            out['usites'].append(q)
        out['crops'], out['ucrops'] = np.stack(out['crops']), np.stack(out['ucrops'])
        return out


def update(student, opt, x, packet, targets, ux, utargets, labels):
    """One optimizer step (the loss of the module docstring). Returns the logged values."""
    ids = packet['ids']
    w, unique = point_weights(np.concatenate(ids), labels)
    offsets = np.cumsum([0] + [len(v) for v in ids])
    st = defaultdict(float)
    opt.zero_grad(set_to_none=True)
    for start in range(0, len(x), MICROBATCH):
        fields = student(x[start:start + MICROBATCH])
        value = fields.sum() * 0
        for k in range(len(fields)):
            j = start + k
            field = fields[k:k + 1]
            q = torch.as_tensor(packet['coordinates'][j], device=x.device)
            y = torch.as_tensor(labels[ids[j]], device=x.device, dtype=torch.float32)
            wj = torch.as_tensor(w[offsets[j]:offsets[j + 1]], device=x.device, dtype=torch.float32)
            exact = (F.binary_cross_entropy_with_logits(voxel_sample(field, q), y, reduction='none') * wj).sum()
            qq, target = targets[j]
            prediction = voxel_sample(field, qq)
            per_site = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
            kd = per_site.mean() / 16
            value = value + (exact + kd * KD)
            st['exact_bce'] += float(exact.detach())
            st['soft_bce'] += float(kd.detach())
            with torch.no_grad():
                prob = prediction.sigmoid()
                kl = per_site - F.binary_cross_entropy(target, target, reduction='none')
                st['distill_kl'] += float(kl.mean()) / 16
                for name, mask in (('low', target < .1), ('middle', (target >= .1) & (target <= .9)), ('high', target > .9)):
                    st[name + '_count'] += int(mask.sum())
                    st[name + '_prob_sum'] += float(prob[mask].sum())
                    st[name + '_kl_sum'] += float(kl[mask].sum())
        assert torch.isfinite(value)
        value.backward()
        del fields, value
    for start in range(0, len(ux), MICROBATCH):
        fields = student(ux[start:start + MICROBATCH])
        value = fields.sum() * 0
        for k in range(len(fields)):
            qq, target = utargets[start + k]
            per_site = F.binary_cross_entropy_with_logits(voxel_sample(fields[k:k + 1], qq), target, reduction='none')
            kd = per_site.mean() / 16
            value = value + UKD * kd
            st['unlabelled_soft_bce'] += float(kd.detach())
            with torch.no_grad():
                entropy = F.binary_cross_entropy(target, target, reduction='none')
                st['unlabelled_kl'] += float((per_site - entropy).mean()) / 16
                st['unlabelled_high'] += int((target > .9).sum())
                st['unlabelled_sites'] += len(target)
        assert torch.isfinite(value)
        value.backward()
        del fields, value
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1, error_if_nonfinite=True)
    opt.step()
    st.update(unique_points=len(unique), gradient_norm=float(norm),
              loss=st['exact_bce'] + KD * st['soft_bce'] + UKD * st['unlabelled_soft_bce'])
    for name in ('low', 'middle', 'high'):
        n = st[name + '_count']
        st[name + '_count'] = int(n)
        prob_sum, kl_sum = st.pop(name + '_prob_sum'), st.pop(name + '_kl_sum')
        st[name + '_student_probability_mean'] = prob_sum / n if n else None
        st[name + '_kl_mean'] = kl_sum / n if n else None
    st['unlabelled_high_fraction'] = st.pop('unlabelled_high') / st.pop('unlabelled_sites')
    return dict(st)


def student_state(state):
    """(network, optimizer, unlabelled rng, unlabelled deck, pending unlabelled plan) of a port train state or an
    original v6c train-state (single arm)."""
    if 'models' in state:
        (arm,) = state['arms']
        assert arm == 'lr2_kd4_u16', f'expected lr2_kd4_u16, got {arm}'
        assert not state['identity'].get('compile', False), 'expected eager training'
        return (state['models'][arm], state['optimizers'][arm], state['unlabelled_rng'], state['unlabelled_deck'],
                state['pending_unlabelled'])
    return state['network'], state['optimizer'], state['unlabelled_rng'], state['unlabelled_deck'], state['pending_unlabelled']


def export(student, out, step):
    path = out / f'step-{step:06d}.safetensors'
    save_weights(student.network, path)
    return path


def complete(out, stage, step, checkpoint, path, **info):
    rt.write_json(out / 'COMPLETE.json', dict(stage=stage, step=step, checkpoint=str(checkpoint), export=str(path),
                                              export_sha256=rt.sha256_file(path), **info))


def run(args):
    assert not (args.stage == 1 and args.init), 'student stage 1 starts from seed 27, not --init'
    out = rt.output_directory(args.out, resume=bool(args.resume))
    rt.configure_torch(SEED)
    args.scribbles = resolve_scribbles(args.scribbles)
    store = ChunkStore(args.ct, cache_gib=args.chunk_cache_gib)
    try:
        train(args, out, store)
    finally:
        store.close()


def train(args, out, store):
    stage, spec = args.stage, STAGES[args.stage]
    teacher_sha256 = rt.sha256_file(args.teacher)
    inputs = dict(input_identity(store, args.scribbles), pool_sha256=rt.sha256_file(args.pool), teacher_sha256=teacher_sha256)
    stop = min(spec['end'], args.stop_at or spec['end'])
    resumed = None
    if args.resume:
        resumed = rt.load(args.resume)
        if resumed.get('stage') != stage:
            raise ValueError(f'{args.resume} is not a stage-{stage} train state')
        rt.check_inputs(resumed, inputs, args.resume)
    step = spec['start'] if resumed is None else resumed['step']
    if step > stop or (step == stop and resumed is None):
        raise ValueError(f'nothing to do: checkpoint step {step}, stop {stop}')
    if step == stop:  # the checkpoint holds the stopping update: no update, only its export and completion marker
        student = Student(seed=SEED)
        student.network.load_state_dict(student_state(resumed)[0], strict=True)
        rt.write_json(out / 'RESUMED.json', dict(stage=stage, start_step=step, stop_step=stop, updates=0,
                                                 resume=args.resume, inputs=inputs, init=resumed.get('init'), time=time.time()))
        path = export(student, out, step)
        if stop == spec['end']:
            complete(out, stage, stop, args.resume, path, repaired=True)
        return
    data = ClickData(store, cache_gib=args.cache_gib, points_json=args.scribbles)
    pool = read_pool(args.pool)
    pool_train = [r['index'] for r in pool if r['split'] == 'train']
    origins = {r['index']: r['origin'] for r in pool}
    pool_ct = Parents(store, lambda i: origins[i], cache_gib=args.cache_gib / 2)

    teacher = Teacher(load_backbone(args.ssl_checkpoint), residual=True, checkpoint_mode='none', seed=SEED)
    teacher_state = rt.load(args.teacher)
    expected_stage, expected_step = ('scribble10k', 10000) if stage == 1 else ('all_aug', 5000)
    assert teacher_state['step'] == expected_step, f'expected {expected_stage} step {expected_step}'
    assert teacher_state.get('stage', teacher_state.get('arm')) == (
        expected_stage if 'stage' in teacher_state else ('scribble' if stage == 1 else 'all_aug'))
    teacher.load_trainable(trainable_of(teacher_state))
    del teacher_state
    teacher.requires_grad_(False).eval()
    student = Student(seed=SEED).train()
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=.01)

    rng = np.random.default_rng(spec['click_seed'])
    deck = sampling.ClassDeck(data.train, data.y, rng)
    urng = np.random.default_rng(UNLABELLED_SEED)
    udeck = sampling.PoolDeck(pool_train, urng)

    def restore(state, clicks=True):
        network, optimizer, unlabelled_rng, unlabelled_deck, pending_unlabelled = student_state(state)
        student.network.load_state_dict(network, strict=True)
        opt.load_state_dict(optimizer)
        rt.set_rng_state(urng, unlabelled_rng)
        udeck.left = copy.deepcopy(unlabelled_deck)
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        if clicks:
            rt.set_rng_state(rng, state['numpy_rng'])
            deck.left = copy.deepcopy(state['decks'])
        return copy.deepcopy(state['pending_plan']), copy.deepcopy(pending_unlabelled)

    init = None
    if resumed is not None:
        pending, upending = restore(resumed)
        init = resumed.get('init')
        del resumed
        export(student, out, step)  # an interruption may have come between the train state and its export
    elif stage == 2:
        assert args.init, '--init (the stage-1 train state at update 20,000) is required for stage 2'
        state = rt.load(args.init)
        assert state['step'] == 20000, state['step']
        if 'stage' in state:
            assert state['stage'] == 1, state['stage']
        _, upending = restore(state, clicks=False)
        del state
        init = dict(path=str(args.init), sha256=rt.sha256_file(args.init))
        # The labelled stream starts anew (seed 2026092612, all 9,627 points); the unlabelled stream continues.
        pending = sampling.draw_cube(rng, deck, data.q, data.parents, data.eval_parents)
    else:
        pending = sampling.draw_cube(rng, deck, data.q, data.parents, data.eval_parents)
        upending = sampling.draw_unlabelled(urng, udeck, origins)
    if args.resume:
        rt.truncate_jsonl(out / 'curve.jsonl', step)
        (out / 'COMPLETE.json').unlink(missing_ok=True)
    rt.write_json(out / ('RESUMED.json' if args.resume else 'STARTED.json'), dict(
        stage=stage, start_step=step, stop_step=stop, end_step=spec['end'], labels=spec['labels'],
        train_points=len(data.train), pool_train_parents=len(pool_train), teacher=str(args.teacher), init=init,
        resume=args.resume, ct=str(args.ct), ssl_checkpoint=str(args.ssl_checkpoint),
        ssl_checkpoint_sha256=SSL_CHECKPOINT_SHA256, teacher_sha256=teacher_sha256, inputs=inputs, time=time.time()))

    look = rt.Lookahead(Loader(data, pool, pool_ct))
    look.submit(pending, upending)
    started = time.time()
    try:
        for number in range(step + 1, stop + 1):
            begin = time.time()
            packet, wait = look.take()
            # Both next plans are drawn before this update (the original order).
            pending = sampling.draw_cube(rng, deck, data.q, data.parents, data.eval_parents)
            upending = sampling.draw_unlabelled(urng, udeck, origins)
            if number < stop:
                look.submit(pending, upending)
            for group in opt.param_groups:
                group['lr'] = learning_rate(stage, number)
            x = views.cube_views(packet['crops'], packet['norm'], [v['photo'] for v in packet['plan']['plans']])
            ux = views.cube_views(packet['ucrops'], packet['unorm'], [v['photo'] for v in packet['uplan']['plans']])
            sites = [torch.as_tensor(s, device=x.device) for s in packet['sites']]
            usites = [torch.as_tensor(s, device=x.device) for s in packet['usites']]
            targets = list(zip(sites, teacher.targets(x, sites)))
            utargets = list(zip(usites, teacher.targets(ux, usites)))
            row = update(student, opt, x, packet, targets, ux, utargets, data.y)
            row = dict(step=number, **row, lr=opt.param_groups[0]['lr'], data_wait=wait, step_seconds=time.time() - begin,
                       seconds=time.time() - started)
            rt.append_jsonl(out / 'curve.jsonl', row)
            del x, ux, targets, utargets, packet
            if number in spec['saves'] or number == stop:
                rt.save(out / f'train-state-{number:06d}.pth', dict(
                    stage=stage, step=number, network=rt.cpu_copy(student.network.state_dict()),
                    optimizer=rt.cpu_copy(opt.state_dict()), numpy_rng=rt.rng_state(rng), decks=copy.deepcopy(deck.left),
                    pending_plan=copy.deepcopy(pending), unlabelled_rng=rt.rng_state(urng),
                    unlabelled_deck=copy.deepcopy(udeck.left), pending_unlabelled=copy.deepcopy(upending),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), labels=spec['labels'],
                    teacher=str(args.teacher), inputs=inputs, init=init))
                export(student, out, number)
        if stop == spec['end']:
            complete(out, stage, stop, out / f'train-state-{stop:06d}.pth', out / f'step-{stop:06d}.safetensors',
                     seconds=time.time() - started)
    finally:
        look.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', type=int, choices=(1, 2), required=True)
    p.add_argument('--ssl-checkpoint', required=True, help='dinovol_2 checkpoint (HF scrollprize/dinovol_v2_ps8_with_paris4_352500)')
    p.add_argument('--teacher', required=True, help='teacher checkpoint: scribble10k step 10,000 (stage 1), all_aug step 5,000 (stage 2)')
    p.add_argument('--out', required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--init', help='stage 2: the stage-1 train state at update 20,000')
    source.add_argument('--resume', help='a train state of this stage written by this script (same inputs)')
    p.add_argument('--scribbles', help='local VC3D ink/noink JSON (default: download released scribbles from Hugging Face)')
    p.add_argument('--pool', default=POOL_CSV, help='unlabelled parent pool CSV')
    p.add_argument('--ct', default=PARIS4_LEVEL0, help='PHercParis4 level-0 zarr array (URL or local directory)')
    p.add_argument('--cache-gib', type=float, default=16., help='decoded parent cache (labelled; half again for the pool)')
    p.add_argument('--chunk-cache-gib', type=float, default=4.)
    p.add_argument('--stop-at', type=int, help='save and stop after this update (smoke tests)')
    run(p.parse_args())


if __name__ == '__main__':
    main()
