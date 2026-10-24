"""Opt-in automatic stop for headless spiral fits ("auto steps").

The stop rule is lasagna's ``steps: "auto"`` rule, copied unchanged from
``lasagna/optimizer.py`` (``_auto_steps_relative_improvement`` and
``_auto_steps_should_stop``): stop once the best loss in the last ``window``
entries of the history improves on the best loss before them by less than
``rel_threshold``, relative to the earlier best. It is copied rather than
imported because spiral-fitting does not depend on the lasagna package, and
it may be adjusted here for spiral fits independently of lasagna.

The fit records one history entry per block of iterations (the lowest loss in
that block), so ``window`` here counts blocks; see fit_spiral.FitContext.run.
"""

import math


def auto_steps_relative_improvement(history: list[float], *, window: int) -> float:
    if len(history) <= int(window):
        return math.inf
    before = history[:-int(window)]
    recent = history[-int(window):]
    if not before or not recent:
        return math.inf
    best_before = min(float(v) for v in before)
    best_recent = min(float(v) for v in recent)
    return (best_before - best_recent) / max(abs(best_before), 1.0e-12)


def auto_steps_should_stop(history: list[float], *, window: int, rel_threshold: float) -> bool:
    return auto_steps_relative_improvement(history, window=window) < float(rel_threshold)
