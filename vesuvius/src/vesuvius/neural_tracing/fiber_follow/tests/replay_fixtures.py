"""Small replay caches in the sole schema; tests state only the fields they exercise."""
import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.data import OnPolicyStates, fiber_manifest
from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, REASON, REPLAY_CLASS


def default_row(n_history, n_future=4):
    nan = np.nan
    return dict(fiber_idx=0, t=0., reverse=False, pos=np.zeros(3), frame=np.eye(3),
                hist=np.zeros((n_history, 3)), hmask=np.zeros(n_history, np.float32),
                seed_pos=np.zeros(3), seed_tangent=np.array([0., 0., 1.]), seed_age=np.float32(0.), seed_valid=False,
                heading_start=0, travelled=0., episode=0, source_row=0,
                supervision=FOLLOWING, supervision_reason=REASON['following'], geometry_valid=True, confidence_valid=True,
                match_distance=np.float32(0.), window_distance=np.float32(0.), match_valid=True, match_ambiguous=False,
                switched=False, beyond_end=False, departure_distance=nan, boundary_distance=nan, switch_distance=nan,
                switch_pos=np.full(3, nan), switch_bank_path='', switch_bank_run='', bad_run=0, bad_run_start=nan,
                would_stop=False, n_commit=4, proposal_points=np.zeros((n_future, 3), np.float32),
                proposal_confidence=np.ones(n_future, np.float32), replay_class=REPLAY_CLASS['ordinary'],
                event_id=-1, hard=False)


def replay_states(fibers, rows, *, track=None, provenance=None, n_history=None):
    """Rows override defaults. Without ``track``, each row's prefix is its own head."""
    n_history = n_history or len(np.asarray(rows[0]['hist']))
    full = []
    for index, row in enumerate(rows):
        value = dict(default_row(n_history), episode=index, source_row=index)
        value.update(row)
        full.append(value)
    if track is None:
        track = np.stack([np.asarray(row['pos'], np.float64) for row in full])
        for index, row in enumerate(full):
            row['seq_start'], row['seq_end'] = index, index+1
    provenance = dict(dict(step=1000, cache_id='fixture', volume=dict(grid_scale=8.)), **(provenance or {}))
    arrays = {key: np.asarray([row[key] for row in full]) for key in OnPolicyStates.FIELDS}
    for key, (dtype, _) in OnPolicyStates.FIELDS.items():
        if dtype != 'U':
            arrays[key] = arrays[key].astype(dtype)
    return OnPolicyStates(manifest=fiber_manifest(fibers), provenance=provenance,
                          track_pos=np.asarray(track, np.float64).reshape(-1, 3), **arrays)
