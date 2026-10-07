"""State-contract fields for hand-built batches."""
import torch

from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, REASON, TERMINAL, UNKNOWN


def state_labels(b):
    """Following states with certified geometry and confidence supervision."""
    return dict(terminal=torch.zeros(b), geometry_valid=torch.ones(b, dtype=torch.bool),
                confidence_valid=torch.ones(b, dtype=torch.bool),
                supervision=torch.full((b,), FOLLOWING, dtype=torch.long),
                supervision_reason=torch.full((b,), REASON['following'], dtype=torch.long),
                match_distance=torch.zeros(b))


def set_terminal(batch, index, reason='switch'):
    batch['terminal'][index] = 1
    batch['geometry_valid'][index] = False
    batch['supervision'][index] = TERMINAL
    batch['supervision_reason'][index] = REASON[reason]


def set_unknown(batch, index, reason='identity'):
    batch['geometry_valid'][index] = False
    batch['confidence_valid'][index] = False
    batch['supervision'][index] = UNKNOWN
    batch['supervision_reason'][index] = REASON[reason]
