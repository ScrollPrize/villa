"""Known-heading roll under competing CT gradients and weak evidence."""
import numpy as np
import pytest

from test_heading import CTOnlyVolume
from vesuvius.neural_tracing.fiber_follow.shared.heading import (
    SeedHeadingError, ct_frame, ct_seed_heading, transverse_frame,
)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import frame_from_heading


def assert_frame(frame, heading):
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(frame[:, 2], heading/np.linalg.norm(heading), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.)


def test_forward_gradient_does_not_hide_transverse_sheet_evidence():
    quality = {}
    frame = transverse_frame(np.diag([2., 1., 10.]), [0., 0., 1.], diagnostics=quality)
    assert_frame(frame, np.array([0., 0., 1.]))
    np.testing.assert_allclose(frame[:, 0], [1., 0., 0.])
    assert quality['source'] == 0 and quality['gap'] == pytest.approx(.5)


def test_measured_paris4_current_crop_tensor_has_stable_transverse_support():
    # Saved CT context: targeted/paris4_targeted_current_001_07.npz.
    # Tensor was measured in ZYX; the production helper accepts XYZ.
    tensor = np.array([
        [.0004909918934622456, 2.0891417632398967e-7, -.0001247266485360563],
        [2.0891417632398967e-7, .0009674068101759095, 7.585016599899958e-6],
        [-.0001247266485360563, 7.585016599899958e-6, .0015452473459899598],
    ])[::-1, ::-1]
    heading = np.array([-.9918323954935153, .023553636378214057, .12535439945593413])
    quality = {}
    frame = transverse_frame(tensor, heading, diagnostics=quality)
    assert_frame(frame, heading)
    assert quality['source'] == 0
    assert quality['gap'] == pytest.approx(.507762467492094)
    # A large, purely forward gradient contributes no transverse orientation.
    stronger = transverse_frame(tensor+.1*np.outer(heading, heading), heading, previous=frame, diagnostics=quality)
    np.testing.assert_allclose(stronger, frame, atol=1e-11)
    assert quality['source'] == 0


@pytest.mark.parametrize('tensor', [np.zeros((3, 3)), np.eye(3),
    np.diag([1., .99, 10.]), np.diag([1e-8, 0., 1.]), np.diag([1e-14, 0., 0.])])
def test_weak_evidence_is_flagged_and_transports_only_when_possible(tensor):
    h = np.array([0., 0., 1.])
    previous = frame_from_heading([.1, 0., 1.], np.array([1., 2., 0.]))
    quality = {}
    frame = transverse_frame(tensor, h, previous, diagnostics=quality)
    assert_frame(frame, h)
    np.testing.assert_allclose(frame, frame_from_heading(h, previous[:, 0]), atol=1e-12)
    assert quality['source'] == 1
    independent = transverse_frame(tensor, h, diagnostics=quality)
    np.testing.assert_array_equal(independent, frame_from_heading(h))
    assert quality['source'] == 2


def test_untransportable_anchor_is_flagged_deterministic():
    quality = {}
    previous = frame_from_heading([1., 0., 0.], np.array([0., 0., 1.]))
    frame = transverse_frame(np.zeros((3, 3)), [0., 0., 1.], previous, diagnostics=quality)
    assert_frame(frame, np.array([0., 0., 1.]))
    assert quality['source'] == 2


def test_transverse_axis_rotates_with_tensor_and_heading():
    rng = np.random.default_rng(20261001)
    for _ in range(50):
        h = rng.normal(size=3)
        a = rng.normal(size=(3, 3))
        tensor = a @ a.T
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        frame = transverse_frame(tensor, h)
        transformed = transverse_frame(rotation @ tensor @ rotation.T, rotation @ h,
                                       previous=rotation @ frame)
        assert_frame(frame, h)
        np.testing.assert_allclose(transformed, rotation @ frame, atol=1e-12)


def test_flat_ct_supplies_roll_but_cannot_supply_initial_seed_heading():
    vol = CTOnlyVolume(np.zeros((97,)*3))
    quality = {}
    frame = ct_frame(vol, [24.]*3, [0., 0., 1.], diagnostics=quality)
    assert_frame(frame, np.array([0., 0., 1.]))
    assert quality['source'] == 2
    with pytest.raises(SeedHeadingError, match='normal'):
        ct_seed_heading(vol, [24.]*3, 'H')


@pytest.mark.parametrize('heading', [[0., 0., 0.], [np.nan, 0., 1.], [0., 1.]])
def test_invalid_heading_remains_an_error(heading):
    with pytest.raises(SeedHeadingError, match='heading'):
        transverse_frame(np.eye(3), heading)


def test_invalid_context_and_io_are_not_hidden_by_roll_fallback(monkeypatch):
    vol = CTOnlyVolume(np.zeros((97,)*3))
    with pytest.raises(SeedHeadingError, match='boundary'):
        ct_frame(vol, [0.]*3, [0., 0., 1.])
    vol.ct.image[40, 40, 40] = np.nan
    with pytest.raises(SeedHeadingError, match='finite'):
        ct_frame(vol, [24.]*3, [0., 0., 1.])
    def unreadable(*args):
        raise OSError('CT read failed')
    monkeypatch.setattr(vol.ct, 'read', unreadable)
    with pytest.raises(OSError, match='CT read failed'):
        ct_frame(vol, [24.]*3, [0., 0., 1.])
