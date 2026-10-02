"""Sample configurations shared by tests."""
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig


def clean_sample(cfg, **overrides):
    """Established simulated traces without excursions, matching a model configuration."""
    return SampleConfig(**dict(dict(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                                    startup_shares=(0., 0., 0., 1.), excursion_probability=0.), **overrides))
