import math
import random

from auto_steps import AutoStepsMonitor, auto_steps_should_stop, auto_steps_stage_starts

BLOCK = 200


def _replay(losses, monitor):
    """Feed per-iteration losses to the monitor as block minima; return the stop
    iteration count, or None when the fit runs to the end."""
    for end in range(BLOCK, len(losses) + 1, BLOCK):
        if monitor.observe(end, min(losses[end - BLOCK:end])):
            return end
    return None


def _monitor(stage_starts):
    # The defaults: a 1,000-iteration window, minimum twice the window.
    return AutoStepsMonitor(window=1000 // BLOCK, block=BLOCK, min_iterations=2000,
                            rel_threshold=1e-4, stage_starts=stage_starts)


def _staged_loss(horizon, stage_start):
    # Flat (fully converged) before the stage; the track DT term then switches
    # on, raising the loss, which decays and levels off again.
    losses = []
    for i in range(horizon):
        value = 10.0 + 0.01 * math.sin(i * 0.37)
        if i >= stage_start:
            value += 5.0 * math.exp(-(i - stage_start) / 300.0) + 2.0
        losses.append(value)
    return losses


def test_stage_starts_keep_only_stages_inside_the_run():
    assert auto_steps_stage_starts([25_001], start_iteration=0, horizon=10_000) == []
    assert auto_steps_stage_starts([25_001], start_iteration=0, horizon=25_001) == []
    assert auto_steps_stage_starts([25_001, 25_001, 5001], start_iteration=0, horizon=30_000) == [5001, 25_001]
    # Resumed past a stage: it already acts from the first resumed iteration.
    assert auto_steps_stage_starts([5001, 20_001], start_iteration=10_000, horizon=30_000) == [20_001]
    assert auto_steps_stage_starts([0], start_iteration=0, horizon=30_000) == []


def test_single_stage_fits_stop_exactly_as_before():
    # Without stages the monitor is the earlier rule: append finite block minima,
    # stop once past the minimum and the window shows no improvement.
    rng = random.Random(7)
    for trial in range(200):
        window = rng.choice([1, 2, 5, 10])
        min_iterations = rng.choice([2 * window * BLOCK, 3000, 7000])
        threshold = rng.choice([0.0, 1e-4, 1e-3, 1e-2])
        scale = rng.uniform(0.5, 5.0)
        values = [100.0 / (1 + n * scale / 10) + rng.uniform(0, 2) for n in range(60)]
        if trial % 10 == 0:
            values[rng.randrange(60)] = float('nan')
        monitor = AutoStepsMonitor(window=window, block=BLOCK, min_iterations=min_iterations,
                                   rel_threshold=threshold)
        history, expected, got = [], None, None
        for n, value in enumerate(values):
            end = (n + 1) * BLOCK
            if math.isfinite(value):
                history.append(value)
            if expected is None and end >= min_iterations and auto_steps_should_stop(
                    history, window=window, rel_threshold=threshold):
                expected = end
            if got is None and monitor.observe(end, value):
                got = end
            if expected is not None and got is not None:
                break
        assert got == expected


def test_staged_fit_runs_past_the_stage_that_the_earlier_rule_stopped_before():
    # A 30,000-step fit whose track DT loss switches on after 25,000 (the
    # reviewer's staged case): flat until then, so the earlier rule stopped it
    # at the first check, long before the DT stage.
    horizon = 30_000
    losses = _staged_loss(horizon, stage_start=25_001)
    single_stage = _monitor([])
    assert _replay(losses, single_stage) == 2000

    monitor = _monitor([25_001])
    assert monitor.stage_starts == [25_001]
    assert monitor.earliest_stop == 27_001
    stop = _replay(losses, monitor)
    # It stops once the DT stage has levelled off, and only on the stage's own losses.
    assert stop is not None and 27_001 <= stop < horizon
    assert min(monitor.history) > max(losses[:25_001])


def test_staged_fit_keeps_running_while_the_new_stage_still_improves():
    horizon = 30_000
    losses = [10.0] * 25_001 + [12.0 - 1e-3 * i for i in range(horizon - 25_001)]
    assert _replay(losses, _monitor([25_001])) is None


def test_block_that_straddles_a_stage_start_is_dropped():
    monitor = AutoStepsMonitor(window=2, block=BLOCK, min_iterations=0, rel_threshold=1e-4,
                               stage_starts=[1001])
    for end in (200, 400, 600, 800, 1000):
        monitor.observe(end, 1.0)
    assert len(monitor.history) == 5
    assert not monitor.observe(1200, 0.5)     # holds iterations before and after the start
    assert monitor.history == []
    monitor.observe(1400, 3.0)
    assert monitor.history == [3.0]
    # A stage starting exactly on a block boundary keeps that block.
    monitor = AutoStepsMonitor(window=2, block=BLOCK, min_iterations=0, rel_threshold=1e-4,
                               stage_starts=[1000])
    for end in (200, 400, 600, 800, 1000, 1200):
        monitor.observe(end, 1.0 if end <= 1000 else 3.0)
    assert monitor.history == [3.0]


def test_resumed_fit_runs_two_windows_before_it_can_stop():
    monitor = AutoStepsMonitor(window=5, block=BLOCK, min_iterations=2000, rel_threshold=1e-4,
                               start_iteration=20_000)
    assert monitor.earliest_stop == 22_000
    stop = next(end for end in range(20_200, 30_001, BLOCK) if monitor.observe(end, 1.0))
    assert stop == 22_000
    # A fresh fit keeps the configured minimum, even below two windows.
    assert AutoStepsMonitor(window=5, block=BLOCK, min_iterations=600,
                            rel_threshold=1e-4).earliest_stop == 600
