"""Shared fixed-state recovery fixtures and rollout evaluation."""
import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.data import (
    OnPolicyStates, fiber_manifest, make_sample, label_state, collate_with_volume,
)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import crop_local_grid
from vesuvius.neural_tracing.fiber_follow.shared.labels import prefix_labels
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.policy import DIAGNOSTIC_THRESHOLDS, select_candidate
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, observed_seed


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
                offtrack=False, hard=True, exploratory=False, drift=drift,
                **observed_seed(item['pos'], item['frame'], item['hist_local'], item['hmask'])))
    if not rows:
        raise ValueError('Recovery fixtures need at least one seed')
    return OnPolicyStates(manifest=fiber_manifest(fibers), provenance=provenance,
        **{k: np.asarray([r[k] for r in rows]) for k in OnPolicyStates.FIELDS+('drift',)+SEED_FIELDS})


@torch.no_grad()
def evaluate_recovery_states(model, vol, states, fibers, sample, *, device='cpu',
                             tracer_class=ModelTracer, batch_builder=None, thresholds=DIAGNOSTIC_THRESHOLDS,
                             recovery_length=32., n_commit=8, tolerance=1.5, sampling_seed=0,
                             limit=0, on_prediction=None, progress=None):
    """Evaluate identical observed states; optional adapters change only model I/O."""
    states.validate_fibers(fibers)
    count = len(states) if limit == 0 else min(limit, len(states))
    if limit < 0 or recovery_length <= 0 or not 1 <= n_commit <= model.cfg.n_future:
        raise ValueError('Invalid recovery evaluation bounds')
    def move(value, *, float_inputs=False):
        if isinstance(value, dict):
            return {k: move(v, float_inputs=float_inputs) for k,v in value.items()}
        dtype = torch.float32 if float_inputs and value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=dtype)
    grid = torch.from_numpy(crop_local_grid(sample.crop)).float()
    rows, predictions = [], []
    for j in range(count):
        fi=int(states.fiber_idx[j]);f=fibers[fi]
        item=label_state(f,states.pos[j],states.frame[j],states.hist[j],states.hmask[j],sample,
                         t=float(states.t[j]),reverse=bool(states.reverse[j]),offtrack=bool(states.offtrack[j]))
        item.update({k: getattr(states, k)[j] for k in SEED_FIELDS if hasattr(states, k)})
        cpu = batch_builder([item], vol) if batch_builder else collate_with_volume([item],vol,sample.crop,grid)
        b = move(cpu)
        sampling={}
        if (getattr(model.cfg,'sampler_mode','zero')=='gaussian'
                or getattr(model.cfg,'gaussian_candidates',0)>0):
            from vesuvius.neural_tracing.fiber_follow.flow_matching.sampling import trace_generator,trace_noise
            generator=trace_generator(sampling_seed,states.pos[j],states.frame[j,:,2])
            sampling['initial_noise']=trace_noise(model.cfg,[generator],device)
        # The original crop builder can return float16, while direct images
        # arrive as a dictionary, including v4's nested remote-seed crop.
        # Floating inputs enter in float32 before AMP; preserve boolean masks.
        images = move(b['x'], float_inputs=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
            output=model(images,b['hist'],b['hmask'],**sampling)
        if on_prediction is not None:
            on_prediction(output, b)
        labels,masks,error=prefix_labels(output['points'],b,tolerance,model.cfg.max_recovery_distance)
        prediction=output['points'][0].float().cpu().numpy();predictions.append(prediction)
        for threshold in thresholds:
            confidence=output['confidence']
            if getattr(model.cfg,'candidate_selection','prefix')=='stop_fallback' and 'candidate_points' in output:
                selected=select_candidate(output['candidate_points'],output['candidate_confidence'],n_commit,
                                          model.cfg.max_recovery_distance,stop_threshold=threshold)
                index=torch.arange(len(selected),device=selected.device)
                points=output['candidate_points'][index,selected]
                confidence=output['candidate_confidence'][index,selected]
                labels,masks,error=prefix_labels(points,b,tolerance,model.cfg.max_recovery_distance)
            tracer=tracer_class(model,vol,sample.crop,sample.n_history,
                TraceParams(max_len=recovery_length,confidence=threshold,seed=sampling_seed,
                            n_commit=n_commit),device=device)
            state={k:getattr(states,k)[j] for k in ('hist','hmask','frame')}
            state.update({k: getattr(states, k)[j] for k in SEED_FIELDS if hasattr(states, k)})
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
                error=float(error[0]),confidence4=float(confidence[0,min(3,model.cfg.n_future-1)]),threshold=threshold,
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
