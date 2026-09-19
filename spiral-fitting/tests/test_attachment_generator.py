"""Every phase bundle component must sample from the generator it is given.

``iter_phase_bundle_losses`` takes a ``generator`` so a caller can hold its
sampling still, and every component of the bundle honours it except
``get_dense_attachment_loss``, which does not accept one and draws from the
process-global RNG instead. It is therefore not held by the seed the caller
chose, and moves whenever anything else on the process has drawn in between.

The scope of that seam is this bundle and no more: ``losses.py`` has no
generator at all. So this is a statement about the phase bundle, not about
training at large, and ``export_preview`` is not the reason for it: previews
are kept off the training stream by the save and restore envelope around them,
not by this argument. The caller that depends on the seam is
``calibration/gradnorm.py``, which seeds one generator per component so the
components can be compared on equal footing.

``get_min_spacing_loss`` honours the generator and is kept here as a control,
so a failure in this file can only mean the attachment loss went to the global
stream.
"""
import math

import torch

from sdt_losses import (
    get_dense_attachment_loss,
    get_min_spacing_loss,
    iter_phase_bundle_losses,
)
from test_sdt_losses import (
    DR_PER_WINDING,
    PerfectSpiralToX,
    sheet_volume,
    spacing_cfg,
)

OUTER_WINDING_IDX = 8
GLOBAL_SEED = 1000
RUN_SEED = 4242


def attachment_inputs():
    """Windings land at x = 1.5 + 3.7 * k across sheets spaced every 10
    voxels, so the loss really does depend on which windings were drawn."""
    volume = sheet_volume(64, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    transform = PerfectSpiralToX(x_offset=1.5, radial_scale=0.37)
    return transform, torch.tensor(DR_PER_WINDING), volume


class GapModel:
    """The least a spiral has to be for min_spacing: a native log gap that
    moves with all three sampled coordinates."""

    device = torch.device('cpu')

    def get_native_log_gaps(self, winding, theta, z):
        return (math.log(5.0)
                + 0.10 * winding.to(torch.float32)
                + 0.20 * torch.sin(theta)
                + 0.01 * z)


def evaluate(measure, *, global_stream_moved):
    """Run ``measure`` from an identical run generator, with the global stream
    either untouched or already advanced by seven draws, which is what any
    interleaved work that samples looks like from here."""
    torch.manual_seed(GLOBAL_SEED)
    if global_stream_moved:
        torch.rand(7)
    return measure(torch.Generator().manual_seed(RUN_SEED))


def attachment_loss(generator):
    transform, dr_per_winding, volume = attachment_inputs()
    cfg = spacing_cfg(sample_count_dense_attachment_points=256)
    loss, _metrics = get_dense_attachment_loss(
        transform, dr_per_winding, volume, OUTER_WINDING_IDX, cfg, 0, 4,
        generator=generator)
    return float(loss)


def min_spacing_loss(generator):
    cfg = spacing_cfg(sample_count_minimum_spacing_independent_samples=256)
    loss, _metrics = get_min_spacing_loss(
        GapModel(), OUTER_WINDING_IDX, cfg, 0, 4, generator=generator)
    return float(loss)


def bundle_components(generator):
    """The real call site, with no normal store so the components that yield
    are min_spacing and dense_attachment."""
    transform, dr_per_winding, volume = attachment_inputs()
    cfg = spacing_cfg(
        sample_count_dense_attachment_points=256,
        sample_count_minimum_spacing_independent_samples=256,
        loss_weight_dense_spacing=0.0,
        loss_weight_dense_spacing_count=0.0,
        loss_weight_min_spacing=1.0,
        loss_weight_dense_attachment=1.0,
    )
    return {
        name: float(loss)
        for name, loss, _metrics in iter_phase_bundle_losses(
            GapModel(), transform, dr_per_winding, volume, None,
            OUTER_WINDING_IDX, cfg, 0, 4,
            generator=generator, with_metrics=False)
    }


def test_min_spacing_is_held_by_the_run_generator():
    """Control: this component passes the generator to every draw it makes."""
    still = evaluate(min_spacing_loss, global_stream_moved=False)
    moved = evaluate(min_spacing_loss, global_stream_moved=True)
    assert still == moved, (
        f'min_spacing moved with the global stream: {still!r} then {moved!r}')


def test_dense_attachment_is_held_by_the_run_generator():
    still = evaluate(attachment_loss, global_stream_moved=False)
    moved = evaluate(attachment_loss, global_stream_moved=True)
    assert still == moved, (
        'dense_attachment moved with the global stream, so it is not drawing '
        f'from the generator it was given: {still!r} then {moved!r}, delta '
        f'{moved - still!r}')


def test_dense_attachment_is_otherwise_deterministic():
    """Guards the case above: with the global stream left alone the loss
    repeats exactly, so any movement seen there is the shift and not float
    noise."""
    first = evaluate(attachment_loss, global_stream_moved=False)
    second = evaluate(attachment_loss, global_stream_moved=False)
    assert first == second, f'{first!r} then {second!r}'


def test_the_phase_bundle_is_held_by_the_run_generator():
    """One generator handed to the bundle, the global stream moved in between,
    and every component it yields must come back the same."""
    still = evaluate(bundle_components, global_stream_moved=False)
    moved = evaluate(bundle_components, global_stream_moved=True)
    assert set(still) == {'min_spacing', 'dense_attachment'}, sorted(still)
    drifted = sorted(name for name in still if still[name] != moved[name])
    assert not drifted, (
        f'the global stream moved these bundle components: {drifted}, with '
        f'the generator identical. Untouched {still}, moved {moved}')
