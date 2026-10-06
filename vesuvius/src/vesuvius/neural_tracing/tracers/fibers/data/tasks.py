"""The per-source task budget: sampling allocations over fresh, live, DAgger and synthetic decisions."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


# One task budget, applied within each dataset source. Sampling allocations, not label
# ratios: every delivered action is labeled by the shared state contract.
TASKS = ('fresh', 'live', 'dagger_pre_excursion', 'dagger_recoverable', 'dagger_terminal',
         'dagger_premature_stop', 'dagger_ordinary', 'synthetic_terminal')
TASK = {name: index for index, name in enumerate(TASKS)}
DEFAULT_TASK_SHARES = dict(fresh=.40, live=.25, dagger_pre_excursion=.08, dagger_recoverable=.06,
                           dagger_terminal=.08, dagger_premature_stop=.03, dagger_ordinary=.05,
                           synthetic_terminal=.05)
FALLBACKS = ('none', 'fresh', 'synthetic', 'no_live_state')
SOURCES = ('fresh', 'live', 'replay', 'synthetic')
SOURCE = {name: index for index, name in enumerate(SOURCES)}
TRAVEL_STRATA = (0., 64., 256., np.inf)


@dataclass(frozen=True)
class TaskBudget:
    """Requested shares per task, plus replay age/reuse limits shared by loader workers.

    ``terminal_fallback_cap`` bounds certified synthetic failures used in place of missing
    real terminal replay, as a fraction of requested terminal slots.
    """
    shares: tuple = tuple(DEFAULT_TASK_SHARES[name] for name in TASKS)
    terminal_fallback_cap: float = .5
    replay_max_age: int = 12000
    replay_event_cap: int = 64

    def __post_init__(self):
        shares = np.asarray(self.shares, np.float64)
        if shares.shape != (len(TASKS),) or not np.isfinite(shares).all() or (shares < 0).any() or abs(shares.sum()-1) > 1e-6:
            raise ValueError(f'Task shares need {len(TASKS)} nonnegative values summing to one ({", ".join(TASKS)})')
        object.__setattr__(self, 'shares', tuple(float(v) for v in shares))
        if not 0 <= self.terminal_fallback_cap <= 1 or self.replay_max_age < 1 or self.replay_event_cap < 1:
            raise ValueError('Invalid terminal fallback cap, replay age ceiling or event cap')

    @classmethod
    def parse(cls, pairs=None, **limits):
        shares = dict(DEFAULT_TASK_SHARES)
        for pair in pairs or ():
            name, _, value = str(pair).partition('=')
            if name not in TASK:
                raise ValueError(f'Unknown task {name!r}; choose from {", ".join(TASKS)}')
            shares[name] = float(value)
        return cls(tuple(shares[name] for name in TASKS), **limits)

    def to_dict(self):
        return dict(shares=dict(zip(TASKS, self.shares)), terminal_fallback_cap=self.terminal_fallback_cap,
                    replay_max_age=self.replay_max_age, replay_event_cap=self.replay_event_cap)


def stable_key(*parts):
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'little') & ((1 << 62)-1)
