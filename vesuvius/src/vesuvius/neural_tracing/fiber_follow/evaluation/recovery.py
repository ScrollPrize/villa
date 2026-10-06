"""Frozen monitor recovery states, separate from calibration and final fibers."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from vesuvius.neural_tracing.fiber_follow.data.data import OnPolicyStates
from vesuvius.neural_tracing.fiber_follow.tracing.heading import FRAME_POLICY
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.evaluation.recovery_fixtures import FIXTURE_STRATA, make_recovery_states, evaluate_recovery_states, recovery_counts
from vesuvius.neural_tracing.fiber_follow.data.observations import observation_builder, FiberTracer
from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostics import decision_rows, summarize_decisions


def monitor_fixture(path, fibers, manifest, sample, spec, seed_count=8):
    path = Path(path)
    seeds = manifest['monitor'][:seed_count]
    if seed_count < 1 or any(s['fiber'] not in manifest['monitor_fibers'] for s in seeds):
        raise ValueError('Recovery diagnostics require monitor seeds')
    provenance = dict(split='monitor', seed_manifest_sha256=manifest['sha256'],
                      volume=spec.to_dict(), sample_cfg=asdict(sample), seeds=seeds,
                      construction='shared fresh sampler: established traces, excursions for displaced strata',
                      strata=[list(s) for s in FIXTURE_STRATA], rng_seed=20260925, frame_policy=FRAME_POLICY,
                      step=-1, cache_id='monitor_recovery')
    # Canonical JSON matches arrays/tuples loaded back from the archive.
    provenance = json.loads(json.dumps(provenance))
    if path.exists():
        states = OnPolicyStates.load(path)
        recorded = json.loads(json.dumps(states.provenance))
        # Fixture states do not depend on the confidence-label tolerance: evaluation relabels every state under the
        # current sample configuration. The CT location (volume spec, the manifest hash covering it, seed headings
        # re-read from it) may move; the states are geometry and evaluation reads the current volume. Settings this
        # code no longer has are ignored.
        sample_keys = set(provenance['sample_cfg'])-{'label_tolerance'}
        unlabeled = lambda value: dict({k: v for k, v in value.items() if k not in ('volume', 'seed_manifest_sha256', 'seeds')},
                                       sample_cfg={k: v for k, v in value.get('sample_cfg', {}).items() if k in sample_keys},
                                       seed_fibers=[seed['fiber'] for seed in value.get('seeds', [])])
        try:
            states.validate_fibers(fibers)
            if unlabeled(recorded) != unlabeled(provenance):
                raise ValueError('its settings differ from this run')
        except ValueError as error:
            previous = path.with_name(path.stem+'.previous.npz')
            print(f'Warning: rebuilding the monitor recovery fixture ({error}); the old one is kept as {previous}',
                  flush=True)
            path.replace(previous)
            return monitor_fixture(path, fibers, manifest, sample, spec, seed_count)
    else:
        states = make_recovery_states(fibers, seeds, sample, provenance, FiberVolume(spec, cache_bytes=256 << 20))
        states.save(path)
        # Use the memory-mapped representation on the first run and after resume alike.
        states = OnPolicyStates.load(path)
    return states, hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_monitor(model, vol, states, fibers, sample, *, device, n_commit=4,
                     tolerance=1.5, recovery_length=32.):
    decisions = []
    thresholds = (.5,)
    rows, _ = evaluate_recovery_states(model, vol, states, fibers, sample, device=device,
        tracer_class=FiberTracer, batch_builder=observation_builder(model.cfg),
        n_commit=n_commit, tolerance=tolerance, thresholds=thresholds, recovery_length=recovery_length,
        on_prediction=lambda out, batch: decisions.extend(decision_rows(out, batch, model.cfg, n_commit, tolerance)))
    return dict(states=len(states), recovery_length=recovery_length,
                decisions=summarize_decisions(decisions, n_commit),
                thresholds={str(t): recovery_counts([r for r in rows if r['threshold'] == t]) for t in thresholds},
                rows=rows)
