# Recovery-aware evaluation

`recovery_scoring.recovery_score` adds annotation-agreement diagnostics without
changing `seeds.score_trace` or its training-monitor behavior; `long_trace_audit`
reports both. Pass the predicted path, `TracedFiber`, frozen start arclength and
direction.

All API length values use trace-grid voxels. On Paris 4 (8 base voxels of 2.4 µm
per trace voxel) multiply by **2** for 9.6 µm reference voxels, or **19.2** for
micrometres.

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
