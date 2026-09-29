# Recovery-aware evaluation

`recovery_scoring.recovery_score` adds annotation-agreement diagnostics without
changing `evaluate.score_trace` or its existing training-monitor behavior.
Pass the same predicted path, `TracedFiber`, frozen start arclength and direction.
Reuse a `PolylineProjector` when scoring several paths against one fiber.

All API length values use trace-grid voxels. For the axial-memory run evaluated
here, multiply by **2** for 9.6 µm reference voxels, or **19.2** for micrometres.

## Definitions

- **Local precision:** supported predicted arclength divided by scored predicted
  arclength. Leaving tolerance affects that portion only; a later return earns
  credit. Predictions of zero scored length have undefined per-path precision.
- **Recovered coverage:** union of forward matched annotation intervals divided
  by available annotation length (capped by the requested evaluation budget).
  Both ends of a credited interval must be within tolerance. Retracing cannot
  count the same interval twice, and a gap between matches is not filled.
- **Correspondence:** exact nearest-polyline-segment distance. A correspondence
  must be reachable from the last supported correspondence within travelled
  arclength plus twice the distance tolerance. Disconnected jumps to nearby
  remote windings cannot establish a new anchor. This conservative geometric
  matching does not independently establish fiber identity.
- **Uninterrupted length:** length preceding the first confirmed excursion.
  If no excursion occurs, the observed length is right-censored; its quantiles
  are not estimates of uncensored failure times. The legacy point-persistence
  score remains separately available and is not renamed or overwritten.
- **Excursions/recoveries:** departure and return each require sustained travel.
  Once confirmed, boundaries are backdated to the start of the respective run.
  Short discrepancies still reduce local precision even without a confirmed
  excursion. Short returns during an active excursion do not end that event.
- **Endpoints:** an unknown annotation endpoint censors subsequent continuation
  after a locally supported endpoint-plane crossing, including after recovery.
  Overrunning a tagged physical endpoint is wrong. A distant plane crossing is
  insufficient to claim an endpoint was reached.

Defaults: tolerance **3 trace voxels / 6 reference voxels**, maximum subdivision
step **0.5 trace voxel / 1 reference voxel**, event persistence **1.5 trace voxels /
3 reference voxels**. Every original prediction vertex is retained. Distances
are exact at samples; tolerance-crossing positions are linearly interpolated
within these small subdivisions. Coverage conservatively credits only intervals
whose sampled endpoints are supported. Persistence is measured in arclength so
it does not change with prediction vertex density.

Additional outputs include recovered supported length, backtracking, unknown
continuation, physical-endpoint overrun, distance mean/quantiles/max, event
locations and durations, and remaining annotation at the final predicted point.
Distance quantiles are weighted by predicted arclength. They exclude censored
unknown tails and known endpoint overruns, which have separate length metrics.

## Full-run artifacts and confidence diagnostics

The runner at `output/axial_memory_seq_run4/eval_all124_warm16_032000/` evaluates
one fixed first start on each of 124 fibers as the priority comparison (372 traces),
then extra starts from the 304-start manifest while time remains, at checkpoint
32,000 EMA, commit8:

- Curved warm16 with annotated incoming direction: confidence 0.5.
- Original frozen point and direction, without supplied history: confidences 0.3 and 0.5.

The warm/cold comparison therefore includes both history and heading changes.
Per-start rows preserve the old scores alongside the new ones. Summaries separate
monitor, calibration and final splits and retain an overall descriptive result.
Paired bootstrap intervals resample whole fibers with their starts intact.

Decision NPZ files retain positions, travelled length, complete proposal curves,
confidence vectors and actual commit counts. Confidence bins in the report use
pointwise proposal proximity at on-annotation heads before the endpoint. This
is a **geometric diagnostic**, not a calibrated target for the model's trained
prefix-confidence objective. Likewise, a flagged potential premature stop means
the first rejected proposal is geometrically near annotation with continuation
remaining; it does not establish that all proposed future steps were safe.

Annotations remain unchanged. In particular, suspected annotation disagreement
on fiber37 is flagged and included, rather than automatically excluded.

## Verification

From `fiber_follow/`, using the existing environment:

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../.. \
  ../../../../.venv/bin/python -m unittest discover -s tests -p test_recovery_scoring.py -v
```

The analytic checks cover straight and sparse annotations, exact projection
against brute force, excursions and recovery, duplicate coverage, reverse
direction, nearby remote windings, unknown/physical endpoints, short deviations,
and empty continuations. Real saved hard-fiber paths additionally check length
partition identities and coverage bounds. Each inference worker checks cached
versus ordinary tracing bit-for-bit; previously saved warm16 hard-case paths are
also compared when encountered.

The user-imposed deadline for this run is 2026-09-29 13:58 UTC. Extra starts
stop being scheduled at 13:50; inference stops at 13:54 so final reports and
primary-start PNGs can finish. Incomplete traces are saved separately and
excluded from paired comparisons, with deferred tasks listed explicitly.
