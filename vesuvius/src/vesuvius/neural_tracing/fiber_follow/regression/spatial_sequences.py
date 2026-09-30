"""Label causal prefixes of observed replay tracks for streaming decisions."""
import numpy as np

from .neighbor_mining import exact_nearest
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, label_state, training_state_allowed
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS


def decision_at(item, builder, j, *, rng=None, observed_indices=None):
    """Label one causal replay prefix without exposing later observations."""
    cfg = builder.cfg
    track = item['memory_track']
    rng = np.random.default_rng(item['identity_seed']) if rng is None else rng
    path = np.asarray(track['pos'][:j+1],float)
    if not np.isfinite(path).all():
        return None
    arc = arclength(path)
    pos,frame = path[-1],np.asarray(track['frame'][j],float)
    behind = arc[-1]-np.arange(1,cfg.n_history+1)
    history = (interp_at(path,arc,np.clip(behind,0,arc[-1])) if len(path) > 1 and arc[-1] > 0
               else np.repeat(pos[None], cfg.n_history, 0))
    fi,current_t,reverse = item['fiber_ref']
    fiber = item.get('supervision_fiber')
    if fiber is None:
        fiber = builder.fibers[fi]
    # Restrict annotation correspondence to the traversed neighborhood; avoid
    # relabeling against a distant return of the same winding fiber.
    remaining = arclength(np.concatenate((track['pos'][j:],np.asarray(item['pos'])[None])))[-1]
    original_t = fiber.length-current_t if reverse else current_t
    extent = 2*remaining+cfg.max_recovery_distance+cfg.n_future*cfg.future_step
    indices = np.flatnonzero((fiber.s >= original_t-extent) & (fiber.s <= original_t+extent))
    if len(indices) < 2:
        return None
    lo,hi = max(0,indices[0]-1),min(len(fiber.s),indices[-1]+2)
    _,_,segment,u = exact_nearest(pos[None],fiber.points[lo:hi])
    k = lo+int(segment[0])
    t = float(fiber.s[k]+u[0]*(fiber.s[k+1]-fiber.s[k]))
    sample = SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future,
                          future_step=cfg.future_step,recent_history_points=cfg.n_history)
    row = label_state(fiber,pos,frame,history,behind >= 0,sample,t=t,reverse=reverse,
                      offtrack=bool(track['offtrack'][j]) if np.isfinite(track['offtrack'][j]) else False)
    observed = slice(None, j) if observed_indices is None else observed_indices
    row.update(fiber_ref=(fi,fiber.length-t if reverse else t,reverse),
               source=item.get('source',2),source_step=item.get('source_step',-1),
               stratum=item.get('stratum',-1),location_source=item.get('location_source',0),
               memory_track={key:np.asarray(value[observed]).copy() for key,value in track.items()})
    for key in ('supervision_fiber','bank_parent_arc_range',*SEED_FIELDS):
        if key in item:
            row[key] = item[key]
    row['seed_age'] = max(0.,float(item.get('seed_age',0.))-remaining)
    builder.prepare(row,fiber,rng)
    band = builder.negative_bank.band if builder.negative_bank is not None else None
    if not training_state_allowed(row,cfg.fine,band) or not builder.footprint_allowed(row,band):
        return None
    return row
