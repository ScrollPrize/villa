"""Frozen monitor recovery states, separate from calibration and final fibers."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from vesuvius.neural_tracing.fiber_follow.data import OnPolicyStates
from vesuvius.neural_tracing.fiber_follow.recovery import make_recovery_states, evaluate_recovery_states, recovery_counts
from vesuvius.neural_tracing.fiber_follow.direct.data import ObservationBuilder, DirectTracer
from vesuvius.neural_tracing.fiber_follow.direct.diagnostics import decision_rows, summarize_decisions


def monitor_fixture(path, fibers, manifest, sample, spec, seed_count=8):
    path = Path(path)
    seeds = manifest['monitor'][:seed_count]
    if seed_count < 1 or any(s['fiber'] not in manifest['monitor_fibers'] for s in seeds):
        raise ValueError('Recovery diagnostics require monitor seeds')
    provenance = dict(split='monitor', seed_manifest_sha256=manifest['sha256'],
                      volume=spec.to_dict(), sample_cfg=asdict(sample), seeds=seeds,
                      construction='frozen augmented states; no confirmed departures', rng_seed=20260925)
    # Canonical JSON matches arrays/tuples loaded back from the archive.
    provenance = json.loads(json.dumps(provenance))
    if path.exists():
        states = OnPolicyStates.load(path)
        states.validate_fibers(fibers)
        if states.provenance != provenance:
            raise ValueError('Monitor recovery fixture settings changed')
    else:
        states = make_recovery_states(fibers, seeds, sample, provenance)
        states.save(path)
        # Replay loading canonicalizes geometry to float32. Use that same
        # representation on the first run and after resume.
        states = OnPolicyStates.load(path)
    return states, hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_monitor(model, vol, states, fibers, sample, *, device, n_commit=4,
                     tolerance=1.5, recovery_length=32.):
    decisions = []
    thresholds = (.5,)
    rows, _ = evaluate_recovery_states(model, vol, states, fibers, sample, device=device,
        tracer_class=DirectTracer, batch_builder=ObservationBuilder(model.cfg),
        n_commit=n_commit, tolerance=tolerance, thresholds=thresholds, recovery_length=recovery_length,
        on_prediction=lambda out, batch: decisions.extend(decision_rows(out, batch, model.cfg, n_commit, tolerance)))
    return dict(states=len(states), recovery_length=recovery_length,
                decisions=summarize_decisions(decisions, n_commit),
                thresholds={str(t): recovery_counts([r for r in rows if r['threshold'] == t]) for t in thresholds},
                rows=rows)
