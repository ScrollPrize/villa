"""Shared fixed-state recovery fixtures and rollout evaluation."""
import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data.data import OnPolicyStates, STARTUP_CATEGORIES, fiber_manifest, make_sample, label_state, replay_facts, resolve_trace_seed
from vesuvius.neural_tracing.fiber_follow.shared.geometry import crop_local_grid
from vesuvius.neural_tracing.fiber_follow.data.labels import prefix_labels
from vesuvius.neural_tracing.fiber_follow.tracing.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DIAGNOSTIC_THRESHOLDS
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS
from vesuvius.neural_tracing.fiber_follow.tracing.heading import FRAME_POLICY, orient_item
from vesuvius.neural_tracing.fiber_follow.data.state_labels import SUPERVISION, replay_class

# Fixture displacement strata (head distance to the original fiber, trace voxels).
FIXTURE_STRATA = ((0., 1.5), (1.5, 3.), (3., 4.5), (4.5, 6.))


def make_recovery_states(fibers, seeds, cfg, provenance, vol, seed=20260925, strata=FIXTURE_STRATA):
    """Freeze one simulated trace state per seed and displacement stratum, preserving identity.

    States come from the shared fresh sampler (established history; displaced strata use its
    smooth excursions) and the shared state contract, with a private RNG.
    """
    rng = np.random.default_rng(seed)
    rows, track = [], []
    established = STARTUP_CATEGORIES.index('established')
    for entry in seeds:
        fi = entry['fiber']
        fiber = fibers[fi]
        reverse = entry['sign'] < 0
        t = fiber.length-entry['t'] if reverse else entry['t']
        for lo, hi in strata:
            for _ in range(1000):
                item = make_sample(fiber, t, reverse, cfg, rng, startup=established, excursion=hi > 1.5)
                if lo <= item['match_distance'] < hi:
                    break
            else:
                raise ValueError('Could not draw a fixture displacement stratum')
            resolve_trace_seed(item, vol)
            orient_item(item, vol)
            facts = item['trace_facts']
            nan = np.nan
            row = dict(fiber_idx=fi, t=float(facts['t']), reverse=reverse, pos=item['pos'], frame=item['frame'],
                       hist=item['hist_local']@item['frame'].T+item['pos'], hmask=item['hmask'],
                       heading_start=0, travelled=float(item['travelled']), episode=len(rows), source_row=0,
                       seq_start=len(track), seq_end=len(track)+len(item['observed_path']),
                       **{k: item[k] for k in SEED_FIELDS},
                       **{k: item[k] for k in ('supervision', 'geometry_valid', 'confidence_valid')},
                       **{k: facts[k] for k in ('match_distance', 'window_distance', 'match_valid', 'match_ambiguous',
                                                'switched', 'beyond_end')},
                       departure_distance=nan, boundary_distance=nan, switch_distance=nan, bad_run=0, bad_run_start=nan,
                       would_stop=False, event_id=-1)
            row['replay_class'] = replay_class(row)
            rows.append(row)
            track.extend(item['observed_path'])
    if not rows:
        raise ValueError('Recovery fixtures need at least one seed')
    return OnPolicyStates(manifest=fiber_manifest(fibers),
        provenance=dict(dict(step=-1, cache_id='monitor_recovery'), **provenance),
        **{k: np.asarray([r[k] for r in rows]) for k in OnPolicyStates.FIELDS},
        track_pos=np.asarray(track, dtype=np.float64))


def recovery_batches(vol, states, fibers, sample, batch_builder=None, limit=0, learned_frames=False):
    """Reconstruct frozen observations identically for images and recovery traces.

    Fixture frames are CT structure-tensor frames. With ``learned_frames`` the recorded frame
    is only the heading prior: the shared builder orients the crop with the model's learned
    heading/normal predictor, exactly as for fresh training states and live traces, and
    relabels targets in that frame.
    """
    grid = torch.from_numpy(crop_local_grid(sample.crop)).float()
    count = len(states) if limit == 0 else min(limit, len(states))
    for j in range(count):
        f = fibers[int(states.fiber_idx[j])]
        item = label_state(f, states.pos[j], states.frame[j], states.hist[j], states.hmask[j], sample,
                          t=float(states.t[j]), reverse=bool(states.reverse[j]), trace=replay_facts(states, j, sample))
        item.update({k: getattr(states, k)[j] for k in SEED_FIELDS if hasattr(states, k)})
        item['observed_path'] = states.observed_prefix(j)
        item['fiber_family'] = f.tag
        if not learned_frames:
            item['frame_policy'] = states.provenance.get('frame_policy', FRAME_POLICY)
        if batch_builder is None:
            raise ValueError('Recovery evaluation requires the common observation builder')
        cpu = batch_builder([item], vol)
        yield j, cpu


@torch.no_grad()
def evaluate_recovery_states(model, vol, states, fibers, sample, *, device='cpu',
                             tracer_class=ModelTracer, batch_builder=None, thresholds=DIAGNOSTIC_THRESHOLDS,
                             recovery_length=32., n_commit=8, tolerance=1.5, sampling_seed=0,
                             limit=0, on_prediction=None, progress=None, learned_frames=None):
    """Evaluate identical observed states; optional adapters change only model I/O.

    ``learned_frames`` (default: whenever the model has a learned frame predictor) orients
    each state's first crop like training and tracing do, instead of the fixture's CT frame.
    """
    from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import frame_predictor
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import LEARNED_FRAME_POLICY
    if learned_frames is None:
        learned_frames = frame_predictor(model.cfg) is not None
    states.validate_fibers(fibers)
    count = len(states) if limit == 0 else min(limit, len(states))
    if limit < 0 or recovery_length <= 0 or not 1 <= n_commit <= model.cfg.n_future:
        raise ValueError('Invalid recovery evaluation bounds')
    if not thresholds:
        raise ValueError('Recovery evaluation needs at least one confidence threshold')
    def move(value, *, float_inputs=False):
        if isinstance(value, dict):
            return {k: move(v, float_inputs=float_inputs) for k,v in value.items()}
        dtype = torch.float32 if float_inputs and value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=dtype)
    rows, predictions = [], []
    for j, cpu in recovery_batches(vol, states, fibers, sample, batch_builder, limit, learned_frames):
        fi=int(states.fiber_idx[j]);f=fibers[fi]
        b = move(cpu)
        sampling={}
        # The original crop builder can return float16, while direct images
        # arrive as a dictionary with independent historical slabs.
        # Floating inputs enter in float32 before AMP; preserve boolean masks.
        images = move(b['x'], float_inputs=True)
        if hasattr(model, 'select_prediction'):
            sampling.update(confidence_threshold=thresholds[0], n_commit=n_commit)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
            output=model(images,b['hist'],b['hmask'],**sampling)
        if on_prediction is not None:
            on_prediction(output, b)
        labels,masks,error=prefix_labels(output['points'],b,tolerance,model.cfg.max_recovery_distance)
        prediction=output['points'][0].float().cpu().numpy();predictions.append(prediction)
        for threshold_index, threshold in enumerate(thresholds):
            confidence=output['confidence']
            if hasattr(model, 'select_prediction'):
                # A different threshold changes when retries stop, not only which
                # prefix is committed. Re-run that policy on the same observations.
                if threshold_index:
                    sampling['confidence_threshold'] = threshold
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                        output = model(images, b['hist'], b['hmask'], **sampling)
                chosen = output
                confidence = chosen['confidence']
                labels, masks, error = prefix_labels(chosen['points'], b, tolerance, model.cfg.max_recovery_distance)
            tracer=tracer_class(model,vol,sample.crop,sample.n_history,
                TraceParams(max_len=recovery_length,confidence=threshold,seed=sampling_seed,
                            n_commit=n_commit),device=device)
            state={k:getattr(states,k)[j] for k in ('hist','hmask','frame')}
            state['observed_path'] = states.observed_prefix(j)
            state['heading_start'] = int(states.heading_start[j])
            state['frame_policy'] = states.provenance.get('frame_policy', FRAME_POLICY)
            if learned_frames:
                # The trace starts from the same oriented crop as the batch above.
                state['frame'] = cpu['crop_frame'][0].numpy()
                state['frame_policy'] = LEARNED_FRAME_POLICY
            state.update({k: getattr(states, k)[j] for k in SEED_FIELDS if hasattr(states, k)})
            try:
                from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import trace_family_kwargs
                paths,reasons=tracer.trace(states.pos[j:j+1],np.asarray(state["frame"])[None,:,2],initial_states=[state],
                                           **trace_family_kwargs(tracer, [f.tag]))
            finally:tracer.close()
            # Match only the original fiber near the stored arc correspondence.
            path=paths[0];sign=-1 if states.reverse[j] else 1
            arc=(f.s-states.t[j])*sign
            near=f.points[(arc>=-8)&(arc<=2.5*recovery_length+32)]
            end_error=float(np.linalg.norm(near-path[-1],axis=-1).min())
            row=dict(state=j,fiber=fi,sampling_seed=sampling_seed,displacement=float(states.match_distance[j]),
                supervision=SUPERVISION[int(states.supervision[j])],
                four_correct=bool(labels[0,min(3,model.cfg.n_future-1)]),four_known=bool(masks[0,min(3,model.cfg.n_future-1)]),first_correct=bool(labels[0,0]),
                first_known=bool(masks[0,0]),would_stop=len(path)==1,commit=max(0,len(path)-1),
                error=float(error[0]),confidence4=float(confidence[0,min(3,model.cfg.n_future-1)]),threshold=threshold,
                end_error=end_error,recovered=bool(len(path)>1 and end_error<=1.5),
                error_reduced=bool(len(path)>1 and end_error<states.match_distance[j]),reason=reasons[0])
            rows.append(row)
        if progress is not None:
            progress(j+1, count)
    return rows, np.asarray(predictions)


def recovery_counts(rows):
    """Keep numerators and denominators by fixture displacement stratum; never average empty ratios."""
    result = {}
    for lo, hi in FIXTURE_STRATA:
        sub = [r for r in rows if lo <= r['displacement'] < hi]
        result[f'{lo:g}-{hi:g}'] = dict(states=len(sub), four_known=sum(r['four_known'] for r in sub),
            four_correct=sum(r['four_known'] and r['four_correct'] for r in sub),
            correct_continuations=sum(r['first_known'] and r['first_correct'] for r in sub),
            false_stops=sum(r['first_known'] and r['first_correct'] and r['would_stop'] for r in sub),
            terminal_continues=sum(r['supervision'] == 'terminal' and not r['would_stop'] for r in sub),
            recovery_observed=sum(r['supervision'] != 'terminal' for r in sub),
            recovered=sum(r.get('recovered', False) and r['supervision'] != 'terminal' for r in sub),
            error_reduced=sum(r.get('error_reduced', False) and r['supervision'] != 'terminal' for r in sub))
    return result
