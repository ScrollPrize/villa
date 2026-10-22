import numpy as np

from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_kollesis import Maps, segment_cells
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_hv_rejection import deposit, comparisons, plots


def inputs():
    d = dict(family=np.array([0, 1]), node_ds=np.array([[0, 1500, 3000], [0, -1500, -3000]]), lengths=np.array([3000., 3000.]))
    u = np.array([[0., 4000.], [500., 4500.]])
    v = np.array([[500., 500.], [3000., 3000.]])
    p = np.array([[.25, .25, .5], [.3, .1, .6]])
    return d, u, v, p


def test_exact_cells_boundaries_reverse_and_count():
    d, u, v, p = inputs()
    maps = Maps((0, 0, 8000, 4000), 1000)
    rows = list(segment_cells(maps, d, u, v, 0))
    assert [len(r[0]) for r in rows] == [3, 3]
    for _, _, _, length in rows:
        np.testing.assert_array_equal(length, [1000, 1000, 1000])
    grids = deposit(d, u, v, p, 1000, 10000)
    heat = grids['map']['heat'].values; profile = grids['map']['profile'].values
    assert heat['v_count'].sum() == 3
    assert profile['v_count'].sum() == 1  # not 3 vertical cell visits
    assert profile['h_count'].sum() == 1
    assert heat['h_disabled_count'].sum() == 0  # strict >.5
    assert heat['v_disabled_count'].sum() == 3
    for mode in grids.values():
        for g in mode.values():
            np.testing.assert_allclose(g.values['h_length'].sum(), 3000, rtol=1e-14)
            np.testing.assert_allclose(g.values['v_length'].sum(), 3000, rtol=1e-14)
    # Conditional counts at V's alternative U coordinate are .75/.25.
    np.testing.assert_allclose(grids['posterior']['heat'].values['v_count'][:, 0].sum(), 2.25)


def test_fraction_difference_ratio_and_weak_denominator():
    g = Maps((0, 0, 0, 0), 1000)
    for family, disabled in (('h', 2.), ('v', 6.)):
        g.values[family+'_count'] = np.array([[10.]])
        g.values[family+'_length'] = np.array([[10000.]])
        g.values[family+'_disabled_count'] = np.array([[disabled]])
        g.values[family+'_disabled_length'] = np.array([[disabled*1000]])
    a = comparisons(g, 'disabled_piece_fraction', {})
    assert a['h'][0, 0] == 20 and a['v'][0, 0] == 60
    assert a['difference'][0, 0] == 40 and a['relative'][0, 0] == 3
    assert np.isnan(comparisons(g, 'disabled_count', {})['relative'][0, 0])  # only 2 disabled H
    g.values['h_disabled_count'][:] = 0
    assert np.isnan(comparisons(g, 'disabled_piece_fraction', {})['relative'][0, 0])


def test_nonzero_origin_zero_endpoint_not_counted_twice():
    d = dict(family=np.array([0, 1]), lengths=np.array([5000., 5000.]),
        node_ds=np.array([[0., 2500., 5000.], [0., -2500., -5000.]]))
    u = np.array([[-5000.], [-5000.]])
    v = np.array([[-5000.], [0.]])
    grid = Maps((-10000., -10000., 10000., 10000.), 10000.)
    for ids, x, y, length in segment_cells(grid, d, u, v, 0):
        assert len(ids) == 1
        assert length[0] == 5000.
        assert x[0] < 0 and y[0] < 0
    profile = Maps((-10000., 0., 10000., 0.), 10000.)
    assert [len(r[0]) for r in segment_cells(profile, d, u, v, 0, profile=True)] == [1, 1]


def test_plot_smoke_and_export(tmp_path):
    from PIL import Image
    d, u, v, p = inputs()
    grids = deposit(d, u, v, p, 1000, 10000)
    plots(grids, tmp_path, {'heatmap_panels': 2, 'minimum_count': .1, 'minimum_length': 1.}, 'Synthetic')
    for mode in ('posterior', 'map'):
        assert (tmp_path/mode/'profiles.svg').exists()
        assert (tmp_path/mode/'disabled_length_heatmaps.svg').exists()
        assert (tmp_path/mode/'profiles.csv').exists()
        with Image.open(tmp_path/mode/'profiles.tif') as image:
            assert image.tag_v2[259] == 5
        with np.load(tmp_path/mode/'profile_raw.npz') as data:
            assert data['step'] == 10000
