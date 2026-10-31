"""Opt-in automatic stop for headless spiral fits ("auto steps").

The stop rule is lasagna's ``steps: "auto"`` rule, copied unchanged from
``lasagna/optimizer.py`` (``_auto_steps_relative_improvement`` and
``_auto_steps_should_stop``): stop once the best loss in the last ``window``
entries of the history improves on the best loss before them by less than
``rel_threshold``, relative to the earlier best. It is copied rather than
imported because spiral-fitting does not depend on the lasagna package, and
it may be adjusted here for spiral fits independently of lasagna.

The fit records one history entry per block of iterations (the lowest loss in
that block), so ``window`` here counts blocks; see AutoStepsMonitor and
fit_spiral.FitContext.run. Unlike lasagna, the monitor restarts the history at
each stage of a staged fit (a loss term switching on part way through).
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


def auto_steps_stage_starts(starts, *, start_iteration, horizon) -> list[int]:
    """The stage starts of a staged fit that the automatic stop must wait for.

    ``starts`` holds the first iteration each later stage acts at (a loss term
    switching on part way through the fit); only those after
    ``start_iteration`` and before ``horizon`` count.
    """
    return sorted({int(s) for s in starts if int(start_iteration) < int(s) < int(horizon)})


class AutoStepsMonitor:
    """The automatic stop applied to a fit's loss, one block at a time.

    The loss is recorded once per ``block`` iterations, as its lowest value in
    that block, and ``window`` counts blocks. A staged fit's loss is not
    comparable across a stage start (a term switching on raises it), so the
    history restarts there, dropping a block that straddles the start, and
    the fit never stops before two windows after the last stage start. A fit
    resumed from a checkpoint (``start_iteration`` > 0) likewise runs two
    windows of its own before it can stop.
    """

    def __init__(self, *, window, block, min_iterations, rel_threshold, stage_starts=(),
                 start_iteration=0):
        self.window = int(window)
        self.block = int(block)
        self.rel_threshold = float(rel_threshold)
        self.stage_starts = sorted(stage_starts)
        self.earliest_stop = int(min_iterations)
        restart = max([*self.stage_starts, int(start_iteration)])
        if restart > 0:
            self.earliest_stop = max(self.earliest_stop, restart + 2 * self.window * self.block)
        self.history = []

    def observe(self, completed, block_min) -> bool:
        """Record the block ending after ``completed`` iterations; True means stop."""
        first = int(completed) - self.block
        started = [s for s in self.stage_starts if first <= s < completed]
        if started:
            self.history.clear()
            if max(started) > first:
                return False
        if math.isfinite(block_min):
            self.history.append(float(block_min))
        return (completed >= self.earliest_stop
                and auto_steps_should_stop(self.history, window=self.window,
                                           rel_threshold=self.rel_threshold))
