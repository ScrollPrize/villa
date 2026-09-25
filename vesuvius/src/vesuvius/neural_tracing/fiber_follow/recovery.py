"""Shared fixed-state recovery fixtures and rollout evaluation."""
import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data import (
    OnPolicyStates, fiber_manifest, make_sample, label_state, collate_with_volume,
)
from vesuvius.neural_tracing.fiber_follow.geometry import crop_local_grid
from vesuvius.neural_tracing.fiber_follow.supervision import prefix_labels
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer, TraceParams


def make_recovery_states(fibers, seeds, cfg, provenance, seed=20260925):
    """Freeze four drift bands per seed using a private RNG, preserving identity."""
    rng = np.random.default_rng(seed)
    rows = []
    for entry in seeds:
        fi = entry['fiber']
        fiber = fibers[fi]
        reverse = entry['sign'] < 0
        t = fiber.length-entry['t'] if reverse else entry['t']
        for lo, hi in ((0., 1.), (1., 1.5), (1.5, 2.), (2., 3.5)):
            for _ in range(1000):
                item = make_sample(fiber, t, reverse, cfg, rng)
                drift = float(np.linalg.norm(item['gt_history'][0]))
                if lo <= drift < hi:
                    break
            else:
                raise ValueError('Could not draw fixture drift band')
            rows.append(dict(fiber_idx=fi, t=entry['t'], reverse=reverse, pos=item['pos'], frame=item['frame'],
                hist=item['hist_local']@item['frame'].T+item['pos'], hmask=item['hmask'],
                offtrack=False, hard=True, exploratory=False, drift=drift))
    if not rows:
        raise ValueError('Recovery fixtures need at least one seed')
    return OnPolicyStates(manifest=fiber_manifest(fibers), provenance=provenance,
        **{k: np.asarray([r[k] for r in rows]) for k in OnPolicyStates.FIELDS+('drift',)})


@torch.no_grad()
def evaluate_recovery_states(model, vol, states, fibers, sample, *, device='cpu',
                             tracer_class=ModelTracer, batch_builder=None, thresholds=(.5, .85),
                             recovery_length=32., n_commit=8, tolerance=1.5, sampling_seed=0,
                             limit=0, archived=False, on_prediction=None, progress=None):
    """Evaluate identical observed states; optional adapters change only model I/O."""
    states.validate_fibers(fibers)
    count = len(states) if limit == 0 else min(limit, len(states))
    if limit < 0 or recovery_length <= 0 or not 1 <= n_commit <= model.cfg.n_future:
        raise ValueError('Invalid recovery evaluation bounds')
    def move(value):
        return {k: move(v) for k,v in value.items()} if isinstance(value, dict) else value.to(device)
    grid = torch.from_numpy(crop_local_grid(sample.crop)).float()
    rows, predictions = [], []
    for j in range(count):
        fi=int(states.fiber_idx[j]);f=fibers[fi]
        item=label_state(f,states.pos[j],states.frame[j],states.hist[j],states.hmask[j],sample,
                         t=float(states.t[j]),reverse=bool(states.reverse[j]),offtrack=bool(states.offtrack[j]))
        cpu = batch_builder([item], vol) if batch_builder else collate_with_volume([item],vol,sample.crop,grid)
        b = move(cpu)
        sampling={}
        if archived:
            model.generator.manual_seed(sampling_seed)
        elif getattr(model.cfg,'sampler_mode','zero')=='gaussian':
            from vesuvius.neural_tracing.fiber_follow.sampling import trace_generator,trace_noise
            generator=trace_generator(sampling_seed,states.pos[j],states.frame[j,:,2])
            sampling['initial_noise']=trace_noise(model.cfg,[generator],device)
        # The original crop builder can return float16, while direct images
        # arrive as a dictionary. Both enter the model in float32 before AMP.
        images = {k: v.float() for k, v in b['x'].items()} if isinstance(b['x'], dict) else b['x'].float()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
            output=model(images,b['hist'],b['hmask'],**sampling)
        if on_prediction is not None:
            on_prediction(output, b)
        labels,masks,error=prefix_labels(output['points'],b,tolerance,model.cfg.max_recovery_distance)
        prediction=output['points'][0].float().cpu().numpy();predictions.append(prediction)
        for threshold in thresholds:
            tracer=tracer_class(model,vol,sample.crop,sample.n_history,
                TraceParams(max_len=recovery_length,confidence=threshold,seed=sampling_seed,
                            n_commit=n_commit),device=device)
            state={k:getattr(states,k)[j] for k in ('hist','hmask','frame')}
            try:
                paths,reasons=tracer.trace(states.pos[j:j+1],states.frame[j:j+1,:,2],initial_states=[state])
            finally:tracer.close()
            # Match only the original fiber near the stored arc correspondence.
            path=paths[0];sign=-1 if states.reverse[j] else 1
            arc=(f.s-states.t[j])*sign
            near=f.points[(arc>=-8)&(arc<=2.5*recovery_length+32)]
            end_error=float(np.linalg.norm(near-path[-1],axis=-1).min())
            row=dict(state=j,fiber=fi,sampling_seed=sampling_seed,drift=float(states.drift[j]),departed=bool(states.offtrack[j]),
                four_correct=bool(labels[0,min(3,model.cfg.n_future-1)]),four_known=bool(masks[0,min(3,model.cfg.n_future-1)]),first_correct=bool(labels[0,0]),
                first_known=bool(masks[0,0]),would_stop=len(path)==1,commit=max(0,len(path)-1),
                error=float(error[0]),confidence4=float(output['confidence'][0,min(3,model.cfg.n_future-1)]),threshold=threshold,
                end_error=end_error,recovered=bool(len(path)>1 and end_error<=1.5),
                error_reduced=bool(len(path)>1 and end_error<states.drift[j]),reason=reasons[0])
            rows.append(row)
        if progress is not None:
            progress(j+1, count)
    return rows, np.asarray(predictions)


def recovery_counts(rows):
    """Keep numerators and denominators; never average empty batch ratios."""
    result={}
    bands=[('<1',0,1),('1-1.5',1,1.5),('1.5-2',1.5,2),('2-3.5',2,3.5),('departed',0,0)]
    for name,lo,hi in bands:
        sub=[r for r in rows if r['departed']] if name=='departed' else [r for r in rows if not r['departed'] and lo<=r['drift']<hi]
        result[name]=dict(states=len(sub),four_known=sum(r['four_known'] for r in sub),
            four_correct=sum(r['four_known'] and r['four_correct'] for r in sub),
            correct_continuations=sum(r['first_known'] and r['first_correct'] for r in sub),
            false_stops=sum(r['first_known'] and r['first_correct'] and r['would_stop'] for r in sub),
            false_continues_after_departure=sum(r['departed'] and not r['would_stop'] for r in sub),
            recovery_observed=sum('recovered' in r and not r['departed'] for r in sub),
            recovered=sum(r.get('recovered',False) and not r['departed'] for r in sub),
            error_reduced=sum(r.get('error_reduced',False) and not r['departed'] for r in sub))
    return result
