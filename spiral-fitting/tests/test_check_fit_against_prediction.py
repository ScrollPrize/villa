"""The spiral-space checks must read 1 winding per layer when the layers sit on the windings, and more when the fit
has extra windings between layers."""
import numpy as np

from check_fit_against_prediction import alignment, empty_windings, layer_centres, summarise, windings_per_layer


def _layers(spacing: float, first: float = 20.0, n: int = 40, per_layer: int = 3000, jitter: float = 0.03, seed: int = 0):
    """Predicted voxels on layers spaced ``spacing`` windings apart, spread over all angles (about 17 per layer per
    2-degree bin, as in a real slice)."""
    rng = np.random.default_rng(seed)
    w = np.concatenate([np.full(per_layer, first + k * spacing) for k in range(n)]) + rng.normal(0, jitter, n * per_layer)
    t = rng.random(len(w))
    return w, t


def test_layers_on_windings():
    w, t = _layers(1.0)
    assert alignment(w) > 0.9
    c = layer_centres(w, t)
    assert abs(windings_per_layer(c) - 1.0) < 0.05
    assert empty_windings(c, 25, 55) < 0.05


def test_extra_windings_between_layers():
    w, t = _layers(1.5)
    c = layer_centres(w, t)
    assert abs(windings_per_layer(c) - 1.5) < 0.05
    assert empty_windings(c, 25, 55) > 0.3          # a third of the windings fall in the gaps
    assert alignment(w) < 0.3                        # half the layers sit half way between windings


def test_summary_has_sectors():
    w, t = _layers(1.0)
    res = summarise(w, t, 25, 55, sectors=12)
    assert len(res["sectors"]) == 12
    assert all(abs(s["windings_per_layer"] - 1.0) < 0.1 for s in res["sectors"])
